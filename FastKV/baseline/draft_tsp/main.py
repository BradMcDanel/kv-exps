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

from baseline.draft_tsp.monkeypatch import replace_llama_for_draft_tsp, set_speculator_data


class DraftTSPPipeline:
    def __init__(self, base_model_name: str, speculator_model_name: str, tokenizer: AutoTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Speculator (Draft) Model...")
        self.speculator_model: PreTrainedModel = self._load_model(speculator_model_name)

        print("Loading Base Model...")
        self.base_model: PreTrainedModel = self._load_model(base_model_name)

        print("Patching for Draft TSP...")
        replace_llama_for_draft_tsp(self.base_model, args)
        
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, types.MethodType] = {}

    def _load_model(self, model_name: str) -> PreTrainedModel:
        """Loads a model with flash_attention_2."""
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2"
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
        attention_scores = torch.stack(all_layer_logits).permute(1, 3, 0, 2, 4)
        bs, num_steps, num_layers, num_heads, key_len = attention_scores.shape
        if bs != 1: raise NotImplementedError("Batch size > 1 is not supported.")

        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(attention_scores.dtype)
        
        if hasattr(self.args, 'kernel_size') and self.args.pooling in ['avgpool', 'maxpool'] and self.args.kernel_size > 1:
            reshaped_for_pooling = attention_probs.squeeze(0).flatten(0, 2)
            
            padding = (self.args.kernel_size - 1) // 2
            pool_fn = F.avg_pool1d if self.args.pooling == 'avgpool' else F.max_pool1d
            
            pooled_tensor = pool_fn(reshaped_for_pooling, kernel_size=self.args.kernel_size, stride=1, padding=padding)
            processed_scores = pooled_tensor.reshape(num_steps, num_layers, num_heads, key_len)
        else:
            processed_scores = attention_probs.squeeze(0)
        
        flattened_scores = processed_scores.flatten(1, 2)
        max_scores = flattened_scores.max(1).values
        final_token_importance = max_scores.mean(0)
        
        # Return the indices sorted by importance
        return torch.argsort(final_token_importance, descending=True)

    def run(self, input_ids: torch.Tensor, max_generation_length: int, look_ahead_k: int) -> Tuple[str, Dict]:
        prompt_len = input_ids.shape[1]
        
        # Get the globally sorted indices
        sorted_indices = self._get_token_importance_scores(input_ids, look_ahead_k)
        # Pass the sorted indices to the monkey-patched model environment
        set_speculator_data(sorted_indices, prompt_len)

        generated_ids = []
        with torch.no_grad():
            prefill_outputs = self.base_model(input_ids=input_ids, use_cache=True)
            past_key_values = prefill_outputs.past_key_values
            
            next_token_logits = prefill_outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids.append(next_token.item())

            # Manual Generation Loop
            for _ in range(max_generation_length - 1):
                cache_len = past_key_values.get_seq_length(0)
                current_token_pos = torch.tensor([[cache_len]], device=self.device, dtype=torch.long)
                
                outputs = self.base_model(
                    input_ids=next_token,
                    position_ids=current_token_pos,
                    cache_position=current_token_pos,
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
    hidden_states = kwargs['hidden_states']
    position_ids = kwargs['position_ids']
    bsz, q_len, _ = hidden_states.size()

    query_states = self_attn.q_proj(hidden_states).view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    key_states = self_attn.k_proj(hidden_states).view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    
    cos, sin = self_attn.rotary_emb(hidden_states, position_ids)
    query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states.detach().clone())

    return pipeline_instance.orig_spec_fwds[self_attn.layer_idx](**kwargs)
