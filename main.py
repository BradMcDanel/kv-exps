import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    repeat_kv as hf_repeat_kv
)
from transformers.cache_utils import Cache, DynamicCache
import types
from typing import List, Dict, Tuple, Optional, Any
import math
import argparse
from datasets import load_dataset
import time

def _speculator_patched_attention_forward_method(
    self_attn: LlamaAttention,
    pipeline_instance: 'SpeculativePrefillPipeline',
    hs: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False, 
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs 
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:

    bsz, q_len, _ = hs.size()

    q_proj = self_attn.q_proj(hs)
    k_proj = self_attn.k_proj(hs)
    v_proj = self_attn.v_proj(hs)

    q = q_proj.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    k_before_rope = k_proj.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    v_for_cache = v_proj.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    
    cos, sin = self_attn.rotary_emb(v_for_cache, position_ids=position_ids)
    q_rope, k_rope = hf_apply_rotary_pos_emb(q, k_before_rope, cos, sin)

    if pipeline_instance.is_generating_lookaheads:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(q_rope.detach().clone())

    kv_out = past_key_value
    if use_cache:
        cache_kwargs = {"cache_position": cache_position}
        # Ensure past_key_value is a Cache object
        if not isinstance(past_key_value, Cache):
            # This case should ideally not happen if use_cache=True implies a Cache object is passed
            # or a new one is created by the model if past_key_value is None.
            # For simplicity, we'll assume it's a Cache object.
            # If it can be None and then a Cache is created internally, that's handled by HF.
            # Our patch assumes if past_key_value is not None and use_cache=True, it's a Cache.
            pass # Or raise an error if it's an unexpected type
        
        k_sdpa, v_sdpa = past_key_value.update(k_rope, v_for_cache, self_attn.layer_idx, cache_kwargs)
    else:
        k_sdpa, v_sdpa = k_rope, v_for_cache
        kv_out = None

    k_sdpa = hf_repeat_kv(k_sdpa, self_attn.num_key_value_groups)
    v_sdpa = hf_repeat_kv(v_sdpa, self_attn.num_key_value_groups)

    attn_mask_in = attention_mask
    is_causal = (q_len > 1) and attn_mask_in is None
    if attn_mask_in is not None:
        actual_kv_seq_len = k_sdpa.shape[-2]
        if attn_mask_in.shape[-1] > actual_kv_seq_len:
            attn_mask_in = attn_mask_in[:, :, :, :actual_kv_seq_len]
        is_causal = False

    attn_output = F.scaled_dot_product_attention(
        q_rope, k_sdpa, v_sdpa, attn_mask=attn_mask_in, dropout_p=0.0, is_causal=is_causal
    )
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self_attn.o_proj.in_features)
    attn_output = self_attn.o_proj(attn_output)
    return attn_output, None, kv_out

class SpeculativePrefillPipeline:
    def __init__(self, base_model_name: str, speculator_model_name: str,
                 attn_implementation: str = "eager", device: str = "cuda",
                 dtype_str: str = "bfloat16", share_kv_cache: bool = False):
        self.base_model_name = base_model_name
        self.speculator_model_name = speculator_model_name
        self.spec_attn_impl_f = "eager"
        self.base_attn_impl = attn_implementation
        self.share_kv_cache = share_kv_cache

        self.device = torch.device("cuda") if device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]
        if self.device.type == "cpu" and self.dtype != torch.float32: self.dtype = torch.float32

        self.tokenizer = self._load_tokenizer()
        self.speculator_model = self._load_model(self.speculator_model_name, self.spec_attn_impl_f)
        self.base_model = self._load_model(self.base_model_name, self.base_attn_impl)
        if self.share_kv_cache: self._check_model_compatibility()
        
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, Any] = {}
        self.is_generating_lookaheads = False

    def _check_model_compatibility(self):
        scfg = self.speculator_model.config; bcfg = self.base_model.config
        compatible = (scfg.num_hidden_layers == bcfg.num_hidden_layers and
                      scfg.hidden_size == bcfg.hidden_size and
                      scfg.num_attention_heads == bcfg.num_attention_heads and
                      getattr(scfg, 'num_key_value_heads', scfg.num_attention_heads) == \
                      getattr(bcfg, 'num_key_value_heads', bcfg.num_attention_heads))
        if not compatible: raise ValueError("Models not compatible for KV cache sharing.")

    def _load_tokenizer(self):
        tok = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        eos_id_val = tok.eos_token_id
        self.eos_token_ids = list(eos_id_val) if isinstance(eos_id_val, list) else ([eos_id_val] if eos_id_val is not None else [])
        eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
        if eot_id != tok.unk_token_id and eot_id not in self.eos_token_ids: self.eos_token_ids.append(eot_id)
        return tok

    def _load_model(self, model_name: str, attn_impl: str):
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        cfg._attn_implementation = attn_impl
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, config=cfg, trust_remote_code=True,
            device_map=self.device, attn_implementation=attn_impl)
        return model.eval()

    def _patch_speculator(self):
        n_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(n_layers)]
        n_patched = 0
        for i, layer in enumerate(self.speculator_model.model.layers):
            if isinstance(layer.self_attn, LlamaAttention):
                attn_mod = layer.self_attn; attn_mod.layer_idx = i
                if i not in self.orig_spec_fwds: self.orig_spec_fwds[i] = attn_mod.forward
                
                # This signature matches LlamaAttention.forward
                # 'self_llama_attn' will be 'attn_mod' when called.
                # 'self' (without suffix) inside this function is the SpeculativePrefillPipeline instance.
                def patched_fwd(
                    self_llama_attn, # This is the LlamaAttention instance (attn_mod)
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_value: Optional[Cache] = None,
                    output_attentions: bool = False,
                    use_cache: bool = False,
                    cache_position: Optional[torch.LongTensor] = None,
                    **inner_kwargs # Catches any other kwargs like 'padding_mask'
                ):
                    return _speculator_patched_attention_forward_method(
                        self_attn=self_llama_attn,
                        pipeline_instance=self, # 'self' of SpeculativePrefillPipeline
                        hs=hidden_states, # Use the explicitly named 'hidden_states'
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **inner_kwargs
                    )
                attn_mod.forward = types.MethodType(patched_fwd, attn_mod)
                n_patched +=1
        return n_patched

    def _convert_tuple_to_dynamic_cache(self, kvt, n_layers):
        if isinstance(kvt, Cache): return kvt
        if kvt is None: return DynamicCache()
        dc = DynamicCache()
        # Ensure kvt has enough elements for the loop, or handle if it's shorter than n_layers
        actual_len = min(len(kvt) if kvt is not None else 0, n_layers)
        dc.key_cache = [kvt[i][0] for i in range(actual_len)] + [None] * (n_layers - actual_len)
        dc.value_cache = [kvt[i][1] for i in range(actual_len)] + [None] * (n_layers - actual_len)
        return dc

    def run(self, prompt_text: str, look_ahead_k: int,
            prompt_keep_percentage: float, max_generation_length: int) -> Tuple[str, Dict[str, float]]:
        
        t_info: Dict[str, float] = {}
        overall_start_t = time.perf_counter()

        n_patched = self._patch_speculator()
        # Ensure max_p_len is positive, though tokenizer handles max_length=0 or negative by not truncating.
        # A very small positive value if calculation results in <=0 is safer.
        max_p_len_calc = self.speculator_model.config.max_position_embeddings - max_generation_length - look_ahead_k - 20
        max_p_len = max(1, max_p_len_calc) 
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=max_p_len).to(self.device)
        p_ids, p_len, bsz = inputs.input_ids, inputs.input_ids.shape[1], inputs.input_ids.shape[0]
        if p_len == 0: # Handle empty prompt after tokenization
            t_info["total_run_time"] = time.perf_counter() - overall_start_t
            return "", t_info


        _st = time.perf_counter()
        self.is_generating_lookaheads = False
        spec_p_cpos = torch.arange(p_len, device=self.device)
        with torch.no_grad():
            out_p_spec = self.speculator_model(input_ids=p_ids, use_cache=True, cache_position=spec_p_cpos)
        spec_kv_full = out_p_spec.past_key_values # This can be Tuple or Cache
        next_tok_spec = torch.argmax(out_p_spec.logits[:, -1, :], dim=-1, keepdim=True)
        for q_list in self.captured_qs: q_list.clear()
        if self.device.type == "cuda": torch.cuda.synchronize()
        t_info["speculation_prefill"] = time.perf_counter() - _st

        _st = time.perf_counter()
        gen_s_ids = []
        # Convert to DynamicCache for the loop, regardless of original type
        cur_kv_spec = self._convert_tuple_to_dynamic_cache(spec_kv_full, self.speculator_model.config.num_hidden_layers)
        
        if n_patched > 0 and look_ahead_k > 0:
            self.is_generating_lookaheads = True
            cur_id_spec, cur_pos_spec = next_tok_spec, p_len
            for _ in range(look_ahead_k):
                s_loop_pids = torch.tensor([[cur_pos_spec]], device=self.device, dtype=torch.long)
                s_loop_cpos = torch.tensor([cur_pos_spec], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    # Pass cur_kv_spec (which is now DynamicCache)
                    out_s = self.speculator_model(input_ids=cur_id_spec, position_ids=s_loop_pids,
                                                  past_key_values=cur_kv_spec, use_cache=True, cache_position=s_loop_cpos)
                cur_kv_spec = out_s.past_key_values # This will be a Cache object from the model
                cur_id_spec = torch.argmax(out_s.logits[:, -1, :], dim=-1, keepdim=True)
                tok_id = cur_id_spec.item()
                gen_s_ids.append(tok_id); cur_pos_spec += 1
                if tok_id in self.eos_token_ids: break
            self.is_generating_lookaheads = False
        if self.device.type == "cuda": torch.cuda.synchronize()
        t_info["speculation_decode"] = time.perf_counter() - _st
        
        _st = time.perf_counter()
        imp_scores = torch.zeros(bsz, p_len, device=self.device, dtype=self.dtype)
        n_la_steps = len(gen_s_ids)
        
        # Importance scoring needs spec_kv_full to be a tuple for direct indexing
        # This assumes speculator with "eager" produces a tuple.
        if n_la_steps > 0 and n_patched > 0 and isinstance(spec_kv_full, tuple) and len(spec_kv_full) > 0:
            # Ensure spec_kv_full is not an empty tuple and layers exist
            example_attn_layer = next((l.self_attn for l in self.speculator_model.model.layers if isinstance(l.self_attn, LlamaAttention)), None)
            if example_attn_layer: # Check if we found a LlamaAttention layer
                h_dim, n_kv_groups = example_attn_layer.head_dim, example_attn_layer.num_key_value_groups
                for lyr_idx in range(self.speculator_model.config.num_hidden_layers):
                    # Add checks for spec_kv_full length and content
                    if lyr_idx >= len(spec_kv_full) or \
                       not (len(self.captured_qs[lyr_idx]) == n_la_steps and \
                            spec_kv_full[lyr_idx] is not None and len(spec_kv_full[lyr_idx]) == 2 and \
                            spec_kv_full[lyr_idx][0] is not None and \
                            spec_kv_full[lyr_idx][0].shape[2] == p_len): 
                        continue
                    k_l = spec_kv_full[lyr_idx][0].detach()
                    k_l_rep = hf_repeat_kv(k_l, n_kv_groups)
                    for spec_idx in range(n_la_steps):
                        q_spec = self.captured_qs[lyr_idx][spec_idx]
                        logits = torch.matmul(q_spec, k_l_rep.transpose(-1, -2)) / math.sqrt(h_dim)
                        imp_scores += logits.sum(dim=1).squeeze(dim=1) # Assumes bsz=1 for squeeze

        n_keep = max(1, math.ceil(p_len * prompt_keep_percentage))
        k_topk = min(n_keep, p_len)
        
        if k_topk > 0 : # Only proceed if there are tokens to select
             _, topk_idx = torch.topk(imp_scores[0], k=k_topk)
             topk_idx_s = torch.sort(topk_idx)[0]
        else: # Handle cases where k_topk is 0 (e.g. p_len is very small)
            topk_idx_s = torch.empty(0, dtype=torch.long, device=self.device)


        if self.share_kv_cache:
            pruned_kv = DynamicCache()
            n_base_layers = self.base_model.config.num_hidden_layers
            pruned_kv.key_cache = [None] * n_base_layers; pruned_kv.value_cache = [None] * n_base_layers
            n_pruned_toks = 0
            # Ensure spec_kv_full is a tuple for this pruning logic.
            if len(topk_idx_s) > 0 and isinstance(spec_kv_full, tuple) and len(spec_kv_full) > 0 : 
                for lyr_idx in range(min(n_base_layers, len(spec_kv_full))): # Iterate safely
                    if spec_kv_full[lyr_idx] is not None and len(spec_kv_full[lyr_idx]) == 2:
                        pruned_kv.key_cache[lyr_idx] = spec_kv_full[lyr_idx][0][:, :, topk_idx_s, :]
                        pruned_kv.value_cache[lyr_idx] = spec_kv_full[lyr_idx][1][:, :, topk_idx_s, :]
                n_pruned_toks = len(topk_idx_s)
            
            ids_ko, pos_ko = p_ids[:, -1:], torch.tensor([[p_len - 1]], device=self.device, dtype=torch.long)
            cpos_ko = torch.tensor([n_pruned_toks], device=self.device, dtype=torch.long)
            with torch.no_grad():
                out_ko = self.base_model(ids_ko, position_ids=pos_ko, past_key_values=pruned_kv, use_cache=True, cache_position=cpos_ko)
            base_next_tok_t, base_kv_dec = torch.argmax(out_ko.logits[:, -1, :], dim=-1, keepdim=True), out_ko.past_key_values
            n_base_kv_toks = base_kv_dec.get_seq_length() if base_kv_dec is not None else 0
        else: # Standard selective prefill
            if len(topk_idx_s) > 0:
                s_ids, s_pos_ids = p_ids[:, topk_idx_s], topk_idx_s.unsqueeze(0)
                prefill_cpos = torch.arange(s_ids.shape[1], device=self.device)
                with torch.no_grad():
                    out_s_prefill = self.base_model(s_ids, position_ids=s_pos_ids, use_cache=True, cache_position=prefill_cpos)
                base_next_tok_t = torch.argmax(out_s_prefill.logits[:, -1, :], dim=-1, keepdim=True)
                base_kv_prefill_out = out_s_prefill.past_key_values
            else: # No tokens selected for prefill
                # Need to generate the first token from scratch or use last token of prompt
                # For simplicity, let's use the last token of the original prompt if available
                # This path needs careful handling if topk_idx_s is empty
                ids_fallback, pos_fallback = p_ids[:, -1:], torch.tensor([[p_len-1]], device=self.device, dtype=torch.long)
                cpos_fallback = torch.tensor([0], device=self.device, dtype=torch.long) # Treat as first token in new cache
                with torch.no_grad():
                     out_fallback = self.base_model(ids_fallback, position_ids=pos_fallback, use_cache=True, cache_position=cpos_fallback)
                base_next_tok_t = torch.argmax(out_fallback.logits[:, -1, :], dim=-1, keepdim=True)
                base_kv_prefill_out = out_fallback.past_key_values


            base_kv_dec = self._convert_tuple_to_dynamic_cache(base_kv_prefill_out, self.base_model.config.num_hidden_layers)
            n_base_kv_toks = base_kv_dec.get_seq_length() if base_kv_dec is not None else 0
        
        if self.device.type == "cuda": torch.cuda.synchronize()
        t_info["base_prefill"] = time.perf_counter() - _st

        _st = time.perf_counter()
        gen_id0 = base_next_tok_t.item(); g_ids = [gen_id0]
        cur_d_id, cur_d_kv = base_next_tok_t, base_kv_dec
        cur_r_pos, cur_c_w_pos = p_len, n_base_kv_toks

        if gen_id0 not in self.eos_token_ids:
            for _ in range(max_generation_length - 1):
                pos_in = torch.tensor([[cur_r_pos]], device=self.device, dtype=torch.long)
                cpos_in_kv = torch.tensor([cur_c_w_pos], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    out_d = self.base_model(cur_d_id, position_ids=pos_in, past_key_values=cur_d_kv, use_cache=True, cache_position=cpos_in_kv)
                next_t_t = torch.argmax(out_d.logits[:, -1, :], dim=-1, keepdim=True)
                next_t_id = next_t_t.item()
                g_ids.append(next_t_id); cur_d_id, cur_d_kv = next_t_t, out_d.past_key_values
                cur_r_pos += 1; cur_c_w_pos +=1
                if next_t_id in self.eos_token_ids: break
        f_text = self.tokenizer.decode(g_ids)
        if self.device.type == "cuda": torch.cuda.synchronize()
        t_info["base_decode"] = time.perf_counter() - _st
        t_info["total_run_time"] = time.perf_counter() - overall_start_t
        return f_text, t_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Prefill Pipeline (Ultra-Simplified)")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--speculator_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa") 
    parser.add_argument("--look_ahead_k", type=int, default=1)
    parser.add_argument("--prompt_keep_percentage", type=float, default=0.2)
    parser.add_argument("--max_generation_length", type=int, default=32)
    parser.add_argument("--base_attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--share_kv_cache", action="store_true", default=False)
    args = parser.parse_args()

    print("--- Initializing Pipeline ---")
    pipeline = SpeculativePrefillPipeline(
        base_model_name=args.base_model_name, speculator_model_name=args.speculator_model_name,
        attn_implementation=args.base_attn_implementation, device=args.device, dtype_str=args.dtype,
        share_kv_cache=args.share_kv_cache)
    print("--- Pipeline Initialized ---")

    if args.dataset_name:
        dataset = load_dataset('THUDM/LongBench', args.dataset_name, split='test')
        sample = dataset[0]
        context_text = sample.get('context', "Default context.")
        input_text = sample.get('input', "Default input.")
        prompt_text_to_run = f"Context: {context_text}\nQuestion: {input_text}\nAnswer:"
    else:
        prompt_text_to_run = "Explain the theory of relativity in simple terms."
    
    print(f"\n--- Running pipeline for prompt (first 100 chars): '{prompt_text_to_run[:100]}...' ---")
    generated_text, run_timing_info = pipeline.run(
        prompt_text=prompt_text_to_run, look_ahead_k=args.look_ahead_k,
        prompt_keep_percentage=args.prompt_keep_percentage, max_generation_length=args.max_generation_length)
    
    print(f"\n--- Generated Output ({len(pipeline.tokenizer.encode(generated_text))} tokens) ---")
    print(f"{generated_text}")

    print(f"\n--- Run Timing Information (seconds) ---")
    for stage, duration in run_timing_info.items():
        print(f"  {stage}: {duration:.4f}")
