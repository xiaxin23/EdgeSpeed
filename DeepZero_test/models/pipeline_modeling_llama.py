from torch import nn
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaPreTrainedModel, GenerationMixin
from typing import Callable, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from abc import ABC, abstractmethod
from typing import List, Dict, Type

def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0


class seqcls_llama_stage(LlamaPreTrainedModel):
    """
    A pipeline stage containing a contiguous subset of transformer layers,
    plus optional embedding or LM head.
    """
    def __init__(self,
                 base_model,
                 layer_start: int,
                 layer_end: int,
                 include_embed: bool,
                 include_lm_head: bool,
                 is_zeroth: bool,
                 last_zeroth_model: bool):
        super().__init__(base_model.config)
        
        self.config = base_model.config
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.include_embed = include_embed
        self.include_lm_head = include_lm_head
        self.is_zeroth = is_zeroth
        self.last_zeroth_model = last_zeroth_model
        
        decoder_modules: list[nn.Module] = []
        
        # Embeddings
        if include_embed:
            self.embed_tokens = base_model.model.embed_tokens
            self.rotary_emb = base_model.model.rotary_emb

        # Transformer layers
        all_layers = base_model.model.layers
        self.layers = nn.ModuleList(all_layers[layer_start:layer_end])
       
        # LM head
        if self.include_lm_head or self.last_zeroth_model:
            self.norm  = base_model.model.norm
            self.score = base_model.score
        self.net = nn.Sequential(*decoder_modules)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor = None,
        causal_mask: torch.Tensor = None,
        position_embeddings: torch.Tensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        if self.include_embed:
            # print("self.embed_tokens: ", self.embed_tokens)
            inputs_embeds = self.embed_tokens(input_ids)
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
            # print("causal_mask: ", causal_mask)
            hidden_states = inputs_embeds
            # print("hidden_states: ", hidden_states)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
            hidden_states = layer_outputs[0]

        if self.include_lm_head:
            hidden_states = self.norm(hidden_states)
            logits = self.score(hidden_states)
            last_non_pad_token = -1
            pooled_logits = logits[torch.arange(hidden_states.shape[0], device=logits.device), last_non_pad_token]
            return logits, pooled_logits
        elif self.last_zeroth_model:
            hidden_states = self.norm(hidden_states)
            logits = self.score(hidden_states)
            last_non_pad_token = -1
            pooled_logits = logits[torch.arange(hidden_states.shape[0], device=logits.device), last_non_pad_token]
            return dict(hidden_states=hidden_states, 
                        causal_mask=causal_mask, 
                        position_embeddings=position_embeddings), logits, pooled_logits
        else:
            return dict(hidden_states=hidden_states, 
                        causal_mask=causal_mask, 
                        position_embeddings=position_embeddings)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        
        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min/2
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min/2
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        
        return causal_mask
  
  
class llama_stage(LlamaPreTrainedModel, GenerationMixin):
    """
    A pipeline stage containing a contiguous subset of transformer layers,
    plus optional embedding or LM head.
    """
    def __init__(self,
                 base_model,
                 layer_start: int,
                 layer_end: int,
                 include_embed: bool,
                 include_lm_head: bool,
                 is_zeroth: bool,
                 last_zeroth_model: bool):
        super().__init__(base_model.config)
        
        self.config = base_model.config
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.include_embed = include_embed
        self.include_lm_head = include_lm_head
        self.is_zeroth = is_zeroth
        self.last_zeroth_model = last_zeroth_model
        
        decoder_modules: list[nn.Module] = []
        
        # Embeddings
        if include_embed:
            self.embed_tokens = base_model.model.embed_tokens
            self.rotary_emb = base_model.model.rotary_emb

        # Transformer layers
        all_layers = base_model.model.layers
        self.layers = nn.ModuleList(all_layers[layer_start:layer_end])
       
        # LM head
        if self.include_lm_head or self.last_zeroth_model:
            self.norm  = base_model.model.norm
            self.lm_head = base_model.lm_head
        self.net = nn.Sequential(*decoder_modules)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor = None,
        causal_mask: torch.Tensor = None,
        position_embeddings: torch.Tensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        if self.include_embed:
            # print("self.embed_tokens: ", self.embed_tokens)
            inputs_embeds = self.embed_tokens(input_ids)
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
            # print("causal_mask: ", causal_mask)
            hidden_states = inputs_embeds
            # print("hidden_states: ", hidden_states)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
            hidden_states = layer_outputs[0]

        if self.include_lm_head:
            hidden_states = self.norm(hidden_states)
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            return logits
        elif self.last_zeroth_model:
            tmp_hidden_states = self.norm(hidden_states)
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(tmp_hidden_states[:, slice_indices, :])
            return dict(hidden_states=hidden_states, 
                        causal_mask=causal_mask, 
                        position_embeddings=position_embeddings), logits
        else:
            return dict(hidden_states=hidden_states, 
                        causal_mask=causal_mask, 
                        position_embeddings=position_embeddings)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        
        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min/2
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min/2
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        
        return causal_mask
    