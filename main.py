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
from typing import List, Dict, Tuple, Optional, Any, Union
import math
import argparse
from datasets import load_dataset
import time
from functools import partial
import sys
import os

# Add QTIP to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'qtip'))

try:
    from lib.linear import QuantizedLinear
    from lib.utils.unsafe_import import model_from_hf_path as qtip_model_from_hf_path
    from model.llama import LlamaAttention as QTIP_LlamaAttention
    QTIP_AVAILABLE = True
except ImportError:
    QTIP_AVAILABLE = False
    print("Warning: QTIP modules not found. QTIP model support disabled.")

def _hf_patched_attention_forward_method(
    self_attn: LlamaAttention,
    pipeline_instance: 'SpeculativePrefillPipeline',
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    batch_size, query_length, _ = hidden_states.size()

    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)

    query_states = query_projection.view(batch_size, query_length, self_attn.config.num_attention_heads, self_attn.head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(batch_size, query_length, self_attn.config.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(batch_size, query_length, self_attn.config.num_key_value_heads, self_attn.head_dim).transpose(1, 2)

    cos, sin = position_embeddings # type: ignore

    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin)

    if pipeline_instance.is_generating_lookaheads:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())

    if use_cache:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(key_states_rotated, value_states_for_cache, self_attn.layer_idx, cache_kwargs) # type: ignore
    else:
        key_states_for_sdpa, value_states_for_sdpa = key_states_rotated, value_states_for_cache

    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, self_attn.num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, self_attn.num_key_value_groups)

    attn_mask_input = attention_mask
    is_sdpa_causal = (query_length > 1) and (attn_mask_input is None)

    if attn_mask_input is not None:
        is_sdpa_causal = False
        actual_kv_sequence_length = key_states_for_sdpa.shape[-2]
        if attn_mask_input.shape[-1] > actual_kv_sequence_length:
            attn_mask_input = attn_mask_input[:, :, :, :actual_kv_sequence_length]

    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, key_states_for_sdpa, value_states_for_sdpa, attn_mask=attn_mask_input, dropout_p=0.0, is_causal=is_sdpa_causal, **kwargs
    )

    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, self_attn.o_proj.in_features)
    attention_output = self_attn.o_proj(attention_output)

    return attention_output, None

def _qtip_patched_attention_forward_method(
    self_attn: LlamaAttention,
    pipeline_instance: 'SpeculativePrefillPipeline',
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Universal attention forward that works with both standard Linear and QuantizedLinear layers."""
    
    batch_size, query_length, _ = hidden_states.size()
    
    # Project Q, K, V - these work whether the projections are nn.Linear or QuantizedLinear
    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)
    
    # Reshape projections
    query_states = query_projection.view(
        batch_size, query_length, self_attn.num_heads, self_attn.head_dim
    ).transpose(1, 2)
    
    key_states_before_rope = key_projection.view(
        batch_size, query_length, self_attn.num_key_value_heads, self_attn.head_dim
    ).transpose(1, 2)
    
    value_states_for_cache = value_projection.view(
        batch_size, query_length, self_attn.num_key_value_heads, self_attn.head_dim
    ).transpose(1, 2)
    
    # Apply rotary embeddings
    cos, sin = position_embeddings
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(
        query_states, key_states_before_rope, cos, sin
    )
    
    # Capture queries if we're generating lookaheads
    if pipeline_instance.is_generating_lookaheads:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(
            query_states_rotated.detach().clone()
        )
    
    # Update cache if needed
    if use_cache:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(
            key_states_rotated, value_states_for_cache, self_attn.layer_idx, cache_kwargs
        )
    else:
        key_states_for_sdpa, value_states_for_sdpa = key_states_rotated, value_states_for_cache
    
    # Repeat KV heads if needed
    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, self_attn.num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, self_attn.num_key_value_groups)
    
    # Prepare attention mask
    attn_mask_input = attention_mask
    is_sdpa_causal = (query_length > 1) and (attn_mask_input is None)
    
    if attn_mask_input is not None:
        is_sdpa_causal = False
        actual_kv_sequence_length = key_states_for_sdpa.shape[-2]
        if attn_mask_input.shape[-1] > actual_kv_sequence_length:
            attn_mask_input = attn_mask_input[:, :, :, :actual_kv_sequence_length]
    
    # Compute attention
    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, 
        key_states_for_sdpa, 
        value_states_for_sdpa, 
        attn_mask=attn_mask_input, 
        dropout_p=0.0, 
        is_causal=is_sdpa_causal, 
        **kwargs
    )
    
    # Reshape and project output
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(
        batch_size, query_length, self_attn.hidden_size
    )
    
    # Output projection - works with both Linear and QuantizedLinear
    attention_output = self_attn.o_proj(attention_output)
    
    return attention_output, None, past_key_value


class SpeculativePrefillPipeline:
    def __init__(self, base_model_name: str, speculator_model_name: Optional[str],
                 share_kv_cache: bool = False):
        self.base_model_name = base_model_name
        self.speculator_model_name = speculator_model_name
        self.share_kv_cache = share_kv_cache
        
        # Load tokenizer first
        self.tokenizer = self._load_tokenizer()
        
        self.speculator_model: Optional[AutoModelForCausalLM] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        
        # Load speculator model if specified
        if self.speculator_model_name is not None and self.speculator_model_name.lower() != "none":
            self.speculator_model = self._load_model(self.speculator_model_name)
            self.device = self.speculator_model.device
            self.dtype = self.speculator_model.dtype
        
        # Load base model
        self.base_model = self._load_model(self.base_model_name)
        self.base_config = self.base_model.config
        
        # Set up EOS token IDs after loading base model
        eos_id_val = self.base_config.eos_token_id
        if isinstance(eos_id_val, int):
            self.eos_token_ids = [eos_id_val]
        elif isinstance(eos_id_val, list):
            self.eos_token_ids = list(eos_id_val)
        elif eos_id_val is None:
            self.eos_token_ids = []
        else:
            self.eos_token_ids = []
        
        if self.device is None:
            self.device = self.base_model.device
            self.dtype = self.base_model.dtype
        
        if self.share_kv_cache and self.speculator_model is not None:
            self._check_model_compatibility()
        
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, Any] = {}
        self.is_generating_lookaheads = False
    
    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        """Load either a standard HF model or a QTIP model based on model name."""
        # Auto-detect QTIP models by checking if they're in the relaxml namespace or have QTIP in name
        is_qtip_model = False
        if QTIP_AVAILABLE and ("relaxml" in model_name.lower() or "qtip" in model_name.lower()):
            is_qtip_model = True
        
        if is_qtip_model:
            print(f"Loading QTIP model: {model_name}")
            model, _ = qtip_model_from_hf_path(model_name, max_mem_ratio=0.7, attn_implementation="eager")
            return model.eval()
        else:
            print(f"Loading standard HF model: {model_name}")
            load_kwargs = {
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "device_map": "auto"
            }
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            return model.eval()
    
    def _check_model_compatibility(self):
        """Check if models are compatible for KV cache sharing."""
        if self.speculator_model is None:
            return
        
        scfg = self.speculator_model.config
        bcfg = self.base_model.config
        
        # Get number of heads, handling different config formats
        spec_num_heads = getattr(scfg, 'num_attention_heads', None)
        base_num_heads = getattr(bcfg, 'num_attention_heads', None)
        spec_num_kv_heads = getattr(scfg, 'num_key_value_heads', spec_num_heads)
        base_num_kv_heads = getattr(bcfg, 'num_key_value_heads', base_num_heads)
        
        compatible = (
            scfg.num_hidden_layers == bcfg.num_hidden_layers and
            scfg.hidden_size == bcfg.hidden_size and
            spec_num_heads == base_num_heads and
            spec_num_kv_heads == base_num_kv_heads
        )
        
        if not compatible:
            raise ValueError("Models not compatible for KV cache sharing.")
    
    def _load_tokenizer(self):
        """Load tokenizer and set up EOS token IDs."""
        try:
            tok = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        except OSError:
            # Fallback for QTIP models that don't include tokenizer files
            # Extract the base model name from QTIP model name
            if "qtip" in self.base_model_name.lower():
                # Try to map to the original model tokenizer
                if "llama-3.1" in self.base_model_name.lower():
                    if "405b" in self.base_model_name.lower():
                        fallback_name = "meta-llama/Llama-3.1-405B-Instruct"
                    elif "70b" in self.base_model_name.lower():
                        fallback_name = "meta-llama/Llama-3.1-70B-Instruct"
                    elif "8b" in self.base_model_name.lower():
                        fallback_name = "meta-llama/Llama-3.1-8B-Instruct"
                    else:
                        fallback_name = "meta-llama/Llama-3.1-8B-Instruct"
                elif "llama-3" in self.base_model_name.lower():
                    if "70b" in self.base_model_name.lower():
                        fallback_name = "meta-llama/Meta-Llama-3-70B-Instruct"
                    elif "8b" in self.base_model_name.lower():
                        fallback_name = "meta-llama/Meta-Llama-3-8B-Instruct"
                    else:
                        fallback_name = "meta-llama/Meta-Llama-3-8B-Instruct"
                elif "llama-2" in self.base_model_name.lower():
                    if "70b" in self.base_model_name.lower():
                        fallback_name = "meta-llama/Llama-2-70b-hf"
                    elif "13b" in self.base_model_name.lower():
                        fallback_name = "meta-llama/Llama-2-13b-hf"
                    elif "7b" in self.base_model_name.lower():
                        fallback_name = "meta-llama/Llama-2-7b-hf"
                    else:
                        fallback_name = "meta-llama/Llama-2-7b-hf"
                else:
                    # Generic Llama fallback
                    fallback_name = "meta-llama/Llama-2-7b-hf"
                
                tok = AutoTokenizer.from_pretrained(fallback_name, trust_remote_code=True)
            else:
                raise
        
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        
        # Set chat template if missing (common for QTIP models)
        if tok.chat_template is None:
            # Determine appropriate template based on model name
            if "llama-3" in self.base_model_name.lower():
                # Llama 3/3.1 template
                tok.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
            elif "llama-2" in self.base_model_name.lower():
                # Llama 2 template
                tok.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
            else:
                # Generic ChatML template as fallback
                tok.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        
        # We'll get the config after loading the base model
        return tok
    
    def _patch_speculator(self):
        """Patch speculator model's attention layers to capture queries."""
        if self.speculator_model is None:
            return 0
        
        if not hasattr(self.speculator_model, 'model') or not hasattr(self.speculator_model.model, 'layers'):
            return 0
        
        num_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        num_patched_layers = 0
        
        for i, layer in enumerate(self.speculator_model.model.layers):
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                attention_module = layer.self_attn
                
                # Store original forward method
                if i not in self.orig_spec_fwds:
                    self.orig_spec_fwds[i] = attention_module.forward
                
                # Apply universal patch that works with both Linear and QuantizedLinear
                partially_applied_func = partial(
                    _hf_patched_attention_forward_method, 
                    pipeline_instance=self
                )
                attention_module.forward = types.MethodType(partially_applied_func, attention_module)
                num_patched_layers += 1
            elif QTIP_AVAILABLE and isinstance(layer.self_attn, QTIP_LlamaAttention):

                # QTIP model - patch with QTIP-specific method
                attention_module = layer.self_attn
                if i not in self.orig_spec_fwds:
                    self.orig_spec_fwds[i] = attention_module.forward
                
                partially_applied_func = partial(
                    _qtip_patched_attention_forward_method, 
                    pipeline_instance=self
                )
                attention_module.forward = types.MethodType(partially_applied_func, attention_module)
                num_patched_layers += 1
        
        return num_patched_layers
    
    def run(self, prompt_text: str, look_ahead_k: int,
            prompt_keep_percentage: float, max_generation_length: int) -> Tuple[str, Dict[str, float]]:
        """Run the speculative prefill pipeline."""
        
        timing_info: Dict[str, float] = {}
        overall_start_time = time.perf_counter()
        
        num_patched_layers = 0
        if self.speculator_model is not None:
            num_patched_layers = self._patch_speculator()
        
        # Calculate max prompt length
        limit_due_to_base = self.base_model.config.max_position_embeddings - max_generation_length
        max_prompt_len_calculated: float
        if self.speculator_model is not None:
            limit_due_to_speculator = self.speculator_model.config.max_position_embeddings - look_ahead_k
            max_prompt_len_calculated = float(min(limit_due_to_base, limit_due_to_speculator) - 20)
        else:
            max_prompt_len_calculated = float(limit_due_to_base - 20)
        max_prompt_length = max(1, int(max_prompt_len_calculated))
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt_text, 
            return_tensors="pt", 
            padding=False, 
            truncation=True, 
            max_length=max_prompt_length
        ).to(self.device)
        
        prompt_input_ids = inputs.input_ids
        prompt_length = inputs.input_ids.shape[1]
        batch_size = inputs.input_ids.shape[0]
        
        if prompt_length == 0:
            timing_info["total_time"] = time.perf_counter() - overall_start_time
            return "", timing_info
        
        # Initialize variables
        speculator_prefill_cache: Optional[Cache] = None
        speculator_prefill_cache_as_tuple: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
        speculator_next_token_ids: Optional[torch.Tensor] = None
        
        # Speculator prefill
        speculation_prefill_time = 0.0
        if self.speculator_model is not None:
            stage_start_time = time.perf_counter()
            self.is_generating_lookaheads = False
            speculator_prompt_cache_position = torch.arange(prompt_length, device=self.device)
            
            with torch.no_grad():
                speculator_prefill_output = self.speculator_model(
                    input_ids=prompt_input_ids, 
                    use_cache=True, 
                    cache_position=speculator_prompt_cache_position
                )
            
            speculator_prefill_cache = speculator_prefill_output.past_key_values
            if speculator_prefill_cache is not None:
                if type(speculator_prefill_cache) is not tuple:
                    speculator_prefill_cache_as_tuple = speculator_prefill_cache.to_legacy_cache()
                else:
                    speculator_prefill_cache_as_tuple = speculator_prefill_cache
                    speculator_prefill_cache = DynamicCache(speculator_prefill_cache)
            
            speculator_next_token_ids = torch.argmax(
                speculator_prefill_output.logits[:, -1, :], 
                dim=-1, 
                keepdim=True
            )
            
            if num_patched_layers > 0:
                for q_list in self.captured_qs:
                    q_list.clear()
            
            speculation_prefill_time = time.perf_counter() - stage_start_time
        
        timing_info["speculation_prefill"] = speculation_prefill_time
        
        # Speculator lookahead generation
        generated_speculator_ids = []
        current_speculator_cache: Optional[Cache] = speculator_prefill_cache
        speculation_decode_time = 0.0
        
        if (self.speculator_model is not None and num_patched_layers > 0 and 
            look_ahead_k > 0 and speculator_next_token_ids is not None and 
            current_speculator_cache is not None):
            
            stage_start_time = time.perf_counter()
            self.is_generating_lookaheads = True
            current_speculator_token_ids = speculator_next_token_ids
            current_speculator_position = prompt_length
            if type(current_speculator_cache) is tuple:
                current_speculator_cache = DynamicCache(current_speculator_cache)
            
            for _ in range(look_ahead_k):
                current_cache_len = current_speculator_cache.get_seq_length(0)
                lookahead_cache_position = torch.tensor([current_cache_len], device=self.device, dtype=torch.long)
                lookahead_position_ids = torch.tensor([[current_speculator_position]], device=self.device, dtype=torch.long)
                
                with torch.no_grad():
                    lookahead_output = self.speculator_model(
                        input_ids=current_speculator_token_ids, 
                        position_ids=lookahead_position_ids,
                        past_key_values=current_speculator_cache, 
                        use_cache=True, 
                        cache_position=lookahead_cache_position
                    )
                
                current_speculator_cache = lookahead_output.past_key_values
                current_speculator_token_ids = torch.argmax(
                    lookahead_output.logits[:, -1, :], 
                    dim=-1, 
                    keepdim=True
                )
                
                token_id = current_speculator_token_ids.item()
                generated_speculator_ids.append(token_id)
                current_speculator_position += 1
                
                if token_id in self.eos_token_ids:
                    break
            
            self.is_generating_lookaheads = False
            speculation_decode_time = time.perf_counter() - stage_start_time
        
        timing_info["speculation_decode"] = speculation_decode_time
        
        # Calculate importance scores
        importance_scores = torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)
        num_lookahead_steps = len(generated_speculator_ids)
        
        if (self.speculator_model is not None and num_lookahead_steps > 0 and 
            num_patched_layers > 0 and speculator_prefill_cache_as_tuple is not None and 
            hasattr(self.speculator_model.model, 'layers')):
            
            example_attn_layer_obj = self.speculator_model.model.layers[0].self_attn
            head_dim = example_attn_layer_obj.head_dim
            num_kv_groups = example_attn_layer_obj.num_key_value_groups
            
            for layer_idx in range(self.speculator_model.config.num_hidden_layers):
                key_layer_prompt = speculator_prefill_cache_as_tuple[layer_idx][0].detach()
                key_layer_prompt_repeated = hf_repeat_kv(key_layer_prompt, num_kv_groups)
                
                for spec_idx in range(num_lookahead_steps):
                    if spec_idx < len(self.captured_qs[layer_idx]):
                        query_speculator_lookahead = self.captured_qs[layer_idx][spec_idx]
                        logits = torch.matmul(
                            query_speculator_lookahead, 
                            key_layer_prompt_repeated.transpose(-1, -2)
                        ) / math.sqrt(head_dim)
                        importance_scores += logits.sum(dim=1).squeeze(dim=1)

        
        # Select important tokens
        sorted_top_k_indices: torch.Tensor
        if (self.speculator_model is not None and num_lookahead_steps > 0 and num_patched_layers > 0):
            num_tokens_to_keep_from_prompt = max(1, math.ceil(prompt_length * prompt_keep_percentage))
            num_top_k_to_select = min(int(num_tokens_to_keep_from_prompt), prompt_length)
            
            if num_top_k_to_select > 0 and importance_scores.sum().item() != 0:
                _, top_k_indices = torch.topk(importance_scores[0], k=num_top_k_to_select)
                sorted_top_k_indices = torch.sort(top_k_indices)[0]
            else:
                sorted_top_k_indices = torch.empty(0, dtype=torch.long, device=self.device)

        else:
            sorted_top_k_indices = torch.empty(0, dtype=torch.long, device=self.device)

        # Base model generation
        base_model_first_token_gen_start_time = time.perf_counter()
        base_model_cache_after_prefill: Optional[Cache] = None
        base_model_next_token_ids: Optional[torch.Tensor] = None
        
        if self.speculator_model is None:
            # No speculator - standard prefill
            base_prefill_cache_position = torch.arange(prompt_length, device=self.device)
            with torch.no_grad():
                base_prefill_output = self.base_model(
                    input_ids=prompt_input_ids, 
                    use_cache=True, 
                    cache_position=base_prefill_cache_position
                )
            base_model_next_token_ids = torch.argmax(
                base_prefill_output.logits[:, -1, :], 
                dim=-1, 
                keepdim=True
            )
            base_model_cache_after_prefill = base_prefill_output.past_key_values
        
        elif self.share_kv_cache:
            # Share KV cache between models
            pruned_kv_cache = DynamicCache()
            n_pruned_tokens_for_cache = 0
            
            if len(sorted_top_k_indices) > 0 and speculator_prefill_cache is not None:
                for layer_idx in range(self.base_model.config.num_hidden_layers):
                    if hasattr(speculator_prefill_cache, 'key_cache') and hasattr(speculator_prefill_cache, 'value_cache'):
                        pruned_key = speculator_prefill_cache.key_cache[layer_idx][:, :, sorted_top_k_indices, :]
                        pruned_value = speculator_prefill_cache.value_cache[layer_idx][:, :, sorted_top_k_indices, :]
                        if pruned_key.dtype != self.base_model.dtype:
                            pruned_key = pruned_key.to(self.base_model.dtype)
                        if pruned_value.dtype != self.base_model.dtype:
                            pruned_value = pruned_value.to(self.base_model.dtype)
                        pruned_kv_cache.update(pruned_key, pruned_value, layer_idx)
                n_pruned_tokens_for_cache = len(sorted_top_k_indices)
            
            knockout_token_ids = prompt_input_ids[:, -1:]
            knockout_position_ids = torch.tensor([[prompt_length - 1]], device=self.device, dtype=torch.long)
            knockout_cache_position = torch.tensor([n_pruned_tokens_for_cache], device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                knockout_output = self.base_model(
                    knockout_token_ids, 
                    position_ids=knockout_position_ids, 
                    past_key_values=pruned_kv_cache, 
                    use_cache=True, 
                    cache_position=knockout_cache_position
                )
            
            base_model_next_token_ids = torch.argmax(
                knockout_output.logits[:, -1, :], 
                dim=-1, 
                keepdim=True
            )
            base_model_cache_after_prefill = knockout_output.past_key_values
        
        else:
            # Not sharing KV cache
            selective_prefill_cache_for_base = DynamicCache()
            base_model_cache_after_selective_prefill: Optional[Cache] = None
            
            if len(sorted_top_k_indices) > 0:
                selected_prompt_ids = prompt_input_ids[:, sorted_top_k_indices]
                selected_position_ids = sorted_top_k_indices.unsqueeze(0)
                selective_prefill_base_cache_position = torch.arange(selected_prompt_ids.shape[1], device=self.device)
                
                with torch.no_grad():
                    selective_prefill_run_output = self.base_model(
                        selected_prompt_ids,
                        position_ids=selected_position_ids,
                        past_key_values=selective_prefill_cache_for_base,
                        use_cache=True,
                        cache_position=selective_prefill_base_cache_position
                    )
                base_model_cache_after_selective_prefill = selective_prefill_run_output.past_key_values
                
                knockout_token_ids = prompt_input_ids[:, -1:]
                knockout_position_ids = torch.tensor([[prompt_length - 1]], device=self.device, dtype=torch.long)
                knockout_cache_position_for_base = torch.tensor([len(sorted_top_k_indices)], device=self.device, dtype=torch.long)
                
                with torch.no_grad():
                    first_token_gen_output = self.base_model(
                        knockout_token_ids,
                        position_ids=knockout_position_ids,
                        past_key_values=base_model_cache_after_selective_prefill,
                        use_cache=True,
                        cache_position=knockout_cache_position_for_base
                    )
                
                base_model_next_token_ids = torch.argmax(
                    first_token_gen_output.logits[:, -1, :], 
                    dim=-1, 
                    keepdim=True
                )
                base_model_cache_after_prefill = first_token_gen_output.past_key_values
            else:
                base_prefill_cache_position = torch.arange(prompt_length, device=self.device)
                with torch.no_grad():
                    base_prefill_output = self.base_model(
                        input_ids=prompt_input_ids,
                        use_cache=True,
                        cache_position=base_prefill_cache_position,
                        past_key_values=selective_prefill_cache_for_base
                    )
                base_model_next_token_ids = torch.argmax(
                    base_prefill_output.logits[:, -1, :], 
                    dim=-1, 
                    keepdim=True
                )
                base_model_cache_after_prefill = base_prefill_output.past_key_values
        
        base_model_first_token_gen_time = time.perf_counter() - base_model_first_token_gen_start_time
        
        timing_info["base_prefill"] = base_model_first_token_gen_time
        if self.speculator_model is not None:
            timing_info["base_ttft"] = speculation_prefill_time + speculation_decode_time + base_model_first_token_gen_time
        else:
            timing_info["base_ttft"] = base_model_first_token_gen_time
        
        # Continue generation
        generated_token_ids: List[int] = []
        final_generated_text = ""
        
        if base_model_next_token_ids is not None and base_model_cache_after_prefill is not None:
            first_generated_token_id = base_model_next_token_ids.item()
            generated_token_ids.append(first_generated_token_id)
            
            current_decode_token_ids = base_model_next_token_ids
            current_decode_cache: Cache = base_model_cache_after_prefill
            
            current_real_position = prompt_length
            if type(current_decode_cache) is not Cache:
                current_decode_cache = DynamicCache(current_decode_cache)
            current_cache_write_position = current_decode_cache.get_seq_length(0)
            
            if first_generated_token_id not in self.eos_token_ids:
                for _ in range(max_generation_length - 1):
                    decode_position_ids = torch.tensor([[current_real_position]], device=self.device, dtype=torch.long)
                    decode_cache_position = torch.tensor([current_cache_write_position], device=self.device, dtype=torch.long)
                    
                    with torch.no_grad():
                        decode_output = self.base_model(
                            current_decode_token_ids, 
                            position_ids=decode_position_ids, 
                            past_key_values=current_decode_cache, 
                            use_cache=True, 
                            cache_position=decode_cache_position
                        )
                    
                    next_base_token_ids = torch.argmax(
                        decode_output.logits[:, -1, :], 
                        dim=-1, 
                        keepdim=True
                    )
                    next_base_token_id = next_base_token_ids.item()
                    generated_token_ids.append(next_base_token_id)
                    
                    current_decode_token_ids = next_base_token_ids
                    current_decode_cache = decode_output.past_key_values
                    
                    current_real_position += 1
                    current_cache_write_position += 1
                    
                    if next_base_token_id in self.eos_token_ids:
                        break
            
            final_generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        timing_info["total_time"] = time.perf_counter() - overall_start_time
        
        # Clean up generated text
        if final_generated_text.startswith("assistant\n\n"):
            final_generated_text = final_generated_text[len("assistant\n\n"):]
        
        return final_generated_text, timing_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Prefill Pipeline with QTIP Support")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--speculator_model_name", type=str, default=None, 
                        help="Path to speculator model. If None or 'None', speculative part is skipped.")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--look_ahead_k", type=int, default=1)
    parser.add_argument("--prompt_keep_percentage", type=float, default=0.2)
    parser.add_argument("--max_generation_length", type=int, default=32)
    parser.add_argument("--share_kv_cache", action="store_true", default=False)
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SpeculativePrefillPipeline(
        base_model_name=args.base_model_name,
        speculator_model_name=args.speculator_model_name,
        share_kv_cache=args.share_kv_cache
    )
    
    # Prepare prompt
    prompt_to_run_str: str
    if args.dataset_name == "hotpotqa":
        dataset = load_dataset('THUDM/LongBench', args.dataset_name, split='test')
        sample = dataset[0]
        context = sample.get('context', '') if isinstance(sample, dict) else ''
        input_text = sample.get('input', '') if isinstance(sample, dict) else ''
        messages = [
            {"role": "user", "content": f"Context: {context}\nQuestion: {input_text}\nAnswer:"}
        ]
        prompt_to_run_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        messages = [
            {"role": "user", "content": "Explain the theory of relativity in simple terms."}
        ]
        prompt_to_run_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if not prompt_to_run_str:
        print("Warning: Generated prompt string is empty.")
        prompt_to_run_str = "Hello!"
    
    # Run pipeline
    generated_text, run_timing_info = pipeline.run(
        prompt_text=prompt_to_run_str,
        look_ahead_k=args.look_ahead_k,
        prompt_keep_percentage=args.prompt_keep_percentage,
        max_generation_length=args.max_generation_length
    )
    
    # Print results
    print(f"\n--- Generated Output ({len(pipeline.tokenizer.encode(generated_text))} tokens) ---")
    print(f"{generated_text}")
    
    print(f"\n--- Run Timing Information (seconds) ---")
    for stage, duration in run_timing_info.items():
        print(f"  {stage}: {duration:.4f}")
    
    # Print model information
    print(f"\n--- Model Information ---")
    print(f"Base model: {args.base_model_name}")
    if args.speculator_model_name:
        print(f"Speculator model: {args.speculator_model_name}")
    else:
        print("No speculator model used")
