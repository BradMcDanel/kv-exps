# baseline/draft_tsp/main.py

import torch
import torch.nn.functional as F
import types
import math
from functools import partial
from typing import List, Dict, Tuple, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

# --- Use the robust HFastKV patching logic ---
from baseline.hfastkv.monkeypatch import replace_llama as replace_llama_for_hfastkv
from baseline.hfastkv.hfastkv_utils import compress as compress_for_hfastkv


class DraftTSPPipeline:
    def __init__(self, base_model_name: str, speculator_model_name: str, tokenizer: AutoTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Speculator (Draft) Model...")
        self.speculator_model: PreTrainedModel = self._load_model(speculator_model_name, patch=False)

        print("Patching for HFastKV and Loading Base Model...")
        # Apply HFastKV patches BEFORE loading the base model to ensure it uses the correct classes.
        replace_llama_for_hfastkv()
        self.base_model: PreTrainedModel = self._load_model(base_model_name, patch=True)

        # Configure the HFastKV-patched base model with the TSP schedule.
        print(f"Configuring base model with HFastKV schedule: '{args.tsp_schedule}'")
        compress_for_hfastkv(self.base_model, args)
        
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, types.MethodType] = {}

    def _load_model(self, model_name: str, patch: bool) -> PreTrainedModel:
        """Loads a model. If patching for HFastKV, attn_implementation must be 'flash_attention_2'."""
        attn_impl = "flash_attention_2" if patch else "sdpa"
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_impl
        ).eval()

    def _patch_speculator(self):
        """Patches the speculator's attention layers to capture Q-vectors for scoring."""
        num_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        for i, layer in enumerate(self.speculator_model.model.layers):
            attn = layer.self_attn
            if i not in self.orig_spec_fwds:
                self.orig_spec_fwds[i] = attn.forward
            attn.forward = types.MethodType(
                partial(_patched_attention_forward_for_capture, pipeline_instance=self), 
                attn
            )

    def _unpatch_speculator(self):
        """Restores the original forward methods on the speculator."""
        for i, layer in enumerate(self.speculator_model.model.layers):
            if i in self.orig_spec_fwds:
                layer.self_attn.forward = self.orig_spec_fwds[i]

    def _get_token_importance_scores(self, prompt_input_ids: torch.Tensor, look_ahead_k: int) -> torch.Tensor:
        """Runs the speculator, generates lookaheads, and computes token importance scores."""
        [q_list.clear() for q_list in self.captured_qs]
        
        with torch.no_grad():
            spec_out = self.speculator_model(input_ids=prompt_input_ids, use_cache=True)
        
        if isinstance(spec_out.past_key_values, tuple):
            spec_prefill_cache_tuple = spec_out.past_key_values
            spec_lookahead_cache = DynamicCache.from_legacy_cache(spec_prefill_cache_tuple)
        else:
            spec_lookahead_cache = spec_out.past_key_values
            spec_prefill_cache_tuple = spec_lookahead_cache.to_legacy_cache()

        self._patch_speculator()
        with torch.no_grad():
            current_tokens = spec_out.logits[:, -1:, :].argmax(-1)
            for i in range(look_ahead_k):
                cache_len = spec_lookahead_cache.get_seq_length(0)
                pos_ids = torch.tensor([[cache_len]], device=self.device, dtype=torch.long)
                outputs = self.speculator_model(input_ids=current_tokens, past_key_values=spec_lookahead_cache, use_cache=True, position_ids=pos_ids)
                
                if isinstance(outputs.past_key_values, tuple): spec_lookahead_cache = DynamicCache.from_legacy_cache(outputs.past_key_values)
                else: spec_lookahead_cache = outputs.past_key_values
                current_tokens = outputs.logits[:, -1:, :].argmax(-1)
        self._unpatch_speculator()

        all_layer_logits = []
        spec_config = self.speculator_model.config
        for layer_idx in range(spec_config.num_hidden_layers):
            if not self.captured_qs[layer_idx] or not spec_prefill_cache_tuple[layer_idx][0].numel(): continue
            
            q_vecs = torch.cat(self.captured_qs[layer_idx], dim=2)
            k_cache = spec_prefill_cache_tuple[layer_idx][0]
            k_cache_rep = repeat_kv(k_cache, spec_config.num_attention_heads // spec_config.num_key_value_heads)
            logits = torch.matmul(q_vecs, k_cache_rep.transpose(2, 3)) / math.sqrt(q_vecs.shape[-1])
            all_layer_logits.append(logits)
        
        if not all_layer_logits: raise RuntimeError("Failed to capture any Q-vectors from the speculator.")
        attention_scores = torch.stack(all_layer_logits).permute(1, 0, 2, 3, 4)
        bs, num_layers, num_heads, num_steps, key_len = attention_scores.shape
        if bs != 1: raise NotImplementedError("Batch size > 1 is not supported.")

        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(attention_scores.dtype)
        reshaped_probs = attention_probs.reshape(bs, -1, key_len)
        if hasattr(self.args, 'kernel_size') and self.args.pooling in ['avgpool', 'maxpool'] and self.args.kernel_size > 1:
            padding = (self.args.kernel_size - 1) // 2
            pool_fn = F.avg_pool1d if self.args.pooling == 'avgpool' else F.max_pool1d
            pooled_tensor = pool_fn(reshaped_probs, kernel_size=self.args.kernel_size, stride=1, padding=padding)
        else:
            pooled_tensor = reshaped_probs
        
        processed_scores = pooled_tensor.reshape(bs, num_layers, num_heads, num_steps, key_len)
        max_over_heads = processed_scores.max(dim=2).values
        mean_over_lookaheads = max_over_heads.mean(dim=2)
        final_token_importance = mean_over_lookaheads.sum(dim=1)
        
        return final_token_importance.squeeze(0)

    def run(self, input_ids: torch.Tensor, max_generation_length: int, look_ahead_k: int) -> Tuple[str, Dict]:
        prompt_len = input_ids.shape[1]
        
        token_scores = self._get_token_importance_scores(input_ids, look_ahead_k)

        if hasattr(self.args, 'initial_capacity_percentage') and self.args.initial_capacity_percentage is not None:
            num_to_select = int(prompt_len * self.args.initial_capacity_percentage)
        else:
            num_to_select = self.args.initial_capacity

        num_to_select = min(num_to_select, prompt_len)
        _, initial_indices = torch.topk(token_scores, k=num_to_select)
        initial_indices, _ = torch.sort(initial_indices)

        selected_input_ids = input_ids[:, initial_indices]
        selected_position_ids = initial_indices.unsqueeze(0)
        
        generated_ids = []
        with torch.no_grad():
            prefill_outputs = self.base_model(input_ids=selected_input_ids, position_ids=selected_position_ids, use_cache=True)
            past_key_values = prefill_outputs.past_key_values
            
            # This is the 1st new token
            next_token_logits = prefill_outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids.append(next_token.item())

            # Manual Generation Loop
            for _ in range(max_generation_length - 1):
                current_token_pos = torch.tensor([[prompt_len + len(generated_ids) - 1]], device=self.device, dtype=torch.long)
                
                outputs = self.base_model(
                    input_ids=next_token,
                    position_ids=current_token_pos,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values

                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token.item())
        
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response, {}

def _patched_attention_forward_for_capture(self_attn: LlamaAttention, pipeline_instance: 'DraftTSPPipeline', **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    """A standard LlamaAttention forward pass that also captures Q-vectors."""
    # We can just call the original forward method and capture the Q-vector.
    # To get the Q-vector, we need to re-compute it just before the original call.
    hidden_states = kwargs['hidden_states']
    position_ids = kwargs['position_ids']
    bsz, q_len, _ = hidden_states.size()

    query_states = self_attn.q_proj(hidden_states).view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    key_states = self_attn.k_proj(hidden_states).view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    
    cos, sin = self_attn.rotary_emb(hidden_states, position_ids)
    query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states.detach().clone())

    # Call the original, unpatched forward method to get the correct output
    return pipeline_instance.orig_spec_fwds[self_attn.layer_idx](**kwargs)
