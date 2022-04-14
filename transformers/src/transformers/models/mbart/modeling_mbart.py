# coding=utf-8
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2021, National Institute of Information and Communication Technology (Raj Dabre)
# Modified portions by Raj Dabre are indicated as so.  
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch MBART model. """
import copy
import math
import random
from typing import Optional, Tuple

import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
## Modified by Raj Dabre. Start.
from torch.autograd import Function
from mixture_of_experts import MoE
## Modified by Raj Dabre. End.

from ...activations import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_mbart import MBartConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MBartConfig"
_TOKENIZER_FOR_DOC = "MBartTokenizer"


MBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mbart-large-cc25",
    # See all MBART models at https://huggingface.co/models?filter=mbart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), -1e10) ## Changed here to -1e10 float("-inf") ## Modified by Raj Dabre.
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


## Modified by Raj Dabre. Start.
# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None, wait_k: Optional[int] = -1, curr_decode_length: Optional[int] = -1):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    if wait_k != -1:
        if curr_decode_length == -1:
            expanded_mask = torch.tril(expanded_mask, wait_k-1) ## This causes the attention mask to be lower triangular to mask future tokens. If wait-k is k then the diagonal shift should be k-1.
        else:
            expanded_mask = torch.tril(expanded_mask, (curr_decode_length-1) + (wait_k-1)) ## This causes the attention mask to be lower triangular to mask future tokens. If wait-k is k then the diagonal shift should be k-1. This is used during decoding time as tgt_len will always be 1 so we need to shift the triangle by an appropriate amount.
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), -1e10) # torch.finfo(dtype).min

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 8,
        hidden_dim = 128,
        activation = ACT2FN['gelu'],
        activation_dropout = 0.0,
        std = 0.2):
        super().__init__()

        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1.normal_(mean=0.0, std=std)
        w2.normal_(mean=0.0, std=std)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation
        self.act_drop = activation_dropout

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = F.dropout(self.act(hidden), p=self.act_drop, training=self.training)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out


class MBartSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)

## Modified by Raj Dabre. End.

# Copied from transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding with Bart->MBart
class MBartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None, "`padding_idx` should not be None, but of type int"
        # MBart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->MBart
class MBartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        multi_source_method = None,
        no_scale_attention_embedding = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5 if not no_scale_attention_embedding else 1.0
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        ## Modified by Raj Dabre. Start.
        if multi_source_method == "merge_after_attention" or multi_source_method == "self_relevance_and_merge_after_attention" or multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only" or multi_source_method == "merge_after_attention_with_context_relevance_only" or multi_source_method == "mid_fusion_merge_after_attention" or multi_source_method == "bottleneck_mid_fusion_merge_after_attention": ## We pass the attentions through a gating method. X and Y are combined as w*x+(1-w)*Y where w=sigmoid(W[X:Y]) where [X:Y] is the concatenation of X and Y along hidden axis.
            if multi_source_method == "merge_after_attention" or multi_source_method == "self_relevance_and_merge_after_attention" or multi_source_method == "mid_fusion_merge_after_attention" or multi_source_method == "bottleneck_mid_fusion_merge_after_attention":
                self.gating_layer = nn.Linear(2*self.head_dim, self.head_dim, bias=False)
            else:
                self.gating_layer = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.multi_source = True
            self.multi_source_method = multi_source_method
        else:
            self.multi_source = False
            self.multi_source_method = ""
        ## Modified by Raj Dabre. End.

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        additional_key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        additional_past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        additional_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        ## Modified by Raj Dabre. Start.
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
            if self.multi_source: # additional_past_key_value is not None
                additional_key_states = additional_past_key_value[0]
                additional_value_states = additional_past_key_value[1]
        ## Modified by Raj Dabre. Enf.
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            ## Modified by Raj Dabre. Start.
            if self.multi_source: # additional_past_key_value is not None
                additional_key_states = self._shape(self.k_proj(additional_key_value_states), -1, bsz)
                additional_value_states = self._shape(self.v_proj(additional_key_value_states), -1, bsz)
            ## Modified by Raj Dabre. End.
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
            ## Modified by Raj Dabre. Start.
            if self.multi_source and is_cross_attention: ## Both conditions are not needed as one multi-source logic can only run when there is cross attention. multi_source is sufficient but keeping this condition for checking.
                additional_past_key_value = (additional_key_states, additional_value_states)
            ## Modified by Raj Dabre. End.
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
        
        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        ## Modified by Raj Dabre. Start.
        if self.multi_source:
            additional_key_states = additional_key_states.view(*proj_shape)
            additional_value_states = additional_value_states.view(*proj_shape)
            additional_src_len = additional_key_states.size(1)
            additional_attn_weights = torch.bmm(query_states, additional_key_states.transpose(1, 2))
            assert additional_attn_weights.size() == (
                bsz * self.num_heads,
                tgt_len,
                additional_src_len,
            ), f"Additional attention weights should be of size {(bsz * self.num_heads, tgt_len, additional_src_len)}, but is {additional_attn_weights.size()}"
            if additional_attention_mask is not None:
                assert additional_attention_mask.size() == (
                    bsz,
                    1,
                    tgt_len,
                    additional_src_len,
                ), f"Attention mask should be of size {(bsz, 1, tgt_len, additional_src_len)}, but is {additional_attention_mask.size()}"
                additional_attn_weights = additional_attn_weights.view(bsz, self.num_heads, tgt_len, additional_src_len) + additional_attention_mask
                additional_attn_weights = additional_attn_weights.view(bsz * self.num_heads, tgt_len, additional_src_len)

            additional_attn_weights = F.softmax(additional_attn_weights, dim=-1)
        ## Modified by Raj Dabre. End.
        
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            ## Modified by Raj Dabre. Start.
            if self.multi_source:
                additional_attn_weights = layer_head_mask.view(1, -1, 1, 1) * additional_attn_weights.view(bsz, self.num_heads, tgt_len, additional_src_len)
                additional_attn_weights = additional_attn_weights.view(bsz * self.num_heads, tgt_len, additional_src_len)
            ## Modified by Raj Dabre. End.
            
        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
            ## Modified by Raj Dabre. Start.
            if self.multi_source:
                additional_attn_weights_reshaped = additional_attn_weights.view(bsz, self.num_heads, tgt_len, additional_src_len)
                additional_attn_weights = additional_attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, additional_src_len)
            ## Modified by Raj Dabre. End.
            
        else:
            attn_weights_reshaped = None
            ## Modified by Raj Dabre. Start.
            if self.multi_source:
                additional_attn_weights_reshaped = None
            ## Modified by Raj Dabre. End.
        
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        
        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
        
        ## Modified by Raj Dabre. Start.
        if self.multi_source:
            additional_attn_probs = F.dropout(additional_attn_weights, p=self.dropout, training=self.training)

            additional_attn_output = torch.bmm(additional_attn_probs, additional_value_states)

            assert additional_attn_output.size() == (
                bsz * self.num_heads,
                tgt_len,
                self.head_dim,
            ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {additional_attn_output.size()}"
            if self.multi_source_method == "merge_after_attention" or self.multi_source_method == "self_relevance_and_merge_after_attention" or self.multi_source_method == "mid_fusion_merge_after_attention" or self.multi_source_method == "bottleneck_mid_fusion_merge_after_attention":
                attentions_merged = torch.cat([attn_output, additional_attn_output], -1) ## Concatenate along hidden axis.
                gating_weight = torch.sigmoid(self.gating_layer(attentions_merged)) ## Compute gating weight.
                attn_output = gating_weight*attn_output + (1.0-gating_weight)*additional_attn_output ## Combine attentions.
            else:
                context_self_relevance_weight = torch.sigmoid(self.gating_layer(additional_attn_output)) ## Compute gating weight.
                attn_output = attn_output + context_self_relevance_weight*additional_attn_output ## Combine attentions.

        ## Modified by Raj Dabre. End.
        
        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)
        
        ## Modified by Raj Dabre. Start.
        if self.multi_source:
            return attn_output, attn_weights_reshaped, additional_attn_weights_reshaped, past_key_value, additional_past_key_value
        else:
            return attn_output, attn_weights_reshaped, past_key_value
        ## Modified by Raj Dabre. End.

class MBartEncoderLayer(nn.Module):
    def __init__(self, config: MBartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.config = config
        moe_loss = () if self.config.use_moe else None
        self.self_attn = MBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            no_scale_attention_embedding=config.no_scale_attention_embedding,
        ) ## An if else condition to either return the sann or a FFT. The FFT will be implemented via a method which pre-generates a bunch of matrices and returns a closure which uses the right matrix during runtime. 
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        if config.use_moe:
            print("Using Mixtures of Experts")
            experts = Experts(dim = self.embed_dim,
                              num_experts = config.num_experts,
                              hidden_dim = config.expert_ffn_size,
                              activation = ACT2FN[config.activation_function],
                              activation_dropout = self.activation_dropout,
                              std = config.init_std)
            self.moe = MoE(
                        dim = self.embed_dim,
                        num_experts = config.num_experts,
                        hidden_dim = config.expert_ffn_size,
                        second_policy_train = 'random',
                        second_policy_eval = 'random',
                        second_threshold_train = 0.2,
                        second_threshold_eval = 0.2,
                        capacity_factor_train = 1.25,
                        capacity_factor_eval = 2.,
                        loss_coef = 1e-2,
                        experts = experts
                    )
        else:
            self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
            self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        adaptor_layers = None,
        deep_adaptor_tuning = False,
        adaptor_layer_idx = 0,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if adaptor_layers is not None and deep_adaptor_tuning: # Apply adaptor layer to current layer's output.
            hidden_states = adaptor_layers(hidden_states, True, adaptor_layer_idx*2)
            
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if self.config.use_moe:
            hidden_states, moe_loss = self.moe(hidden_states)
        else:
            hidden_states = self.activation_fn(self.fc1(hidden_states))
            hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
            hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if adaptor_layers is not None and deep_adaptor_tuning: # Apply adaptor layer to current layer's output.
            hidden_states = adaptor_layers(hidden_states, True, adaptor_layer_idx*2+1)

        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        if self.config.use_moe:
            outputs = ([hidden_states, moe_loss],)
        else:
            outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MBartDecoderLayer(nn.Module):
    def __init__(self, config: MBartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.config = config
        self.self_attn = MBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            no_scale_attention_embedding=config.no_scale_attention_embedding,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = MBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            multi_source_method=config.multi_source_method,
            no_scale_attention_embedding=config.no_scale_attention_embedding,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        if config.use_moe:
            print("Using Mixtures of Experts")
            experts = Experts(dim = self.embed_dim,
                              num_experts = config.num_experts,
                              hidden_dim = config.expert_ffn_size,
                              activation = ACT2FN[config.activation_function],
                              activation_dropout = self.activation_dropout,
                              std = config.init_std)
            self.moe = MoE(
                        dim = self.embed_dim,
                        num_experts = config.num_experts,
                        hidden_dim = config.expert_ffn_size,
                        second_policy_train = 'random',
                        second_policy_eval = 'random',
                        second_threshold_train = 0.2,
                        second_threshold_eval = 0.2,
                        capacity_factor_train = 1.25,
                        capacity_factor_eval = 2.,
                        loss_coef = 1e-2,
                        experts = experts
                    )
        else:
            self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
            self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        encoder_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        additional_encoder_hidden_states: Optional[torch.Tensor] = None,
        additional_encoder_attention_mask: Optional[torch.Tensor] = None,
        adaptor_layers = None,
        deep_adaptor_tuning = False,
        adaptor_layer_idx = 0,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        #print(attention_mask.size() if attention_mask is not None else 1, encoder_attention_mask.size() if encoder_attention_mask is not None else 1, additional_encoder_attention_mask.size() if additional_encoder_attention_mask is not None else 1)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if adaptor_layers is not None and deep_adaptor_tuning: # Apply adaptor layer to current layer's output.
            hidden_states = adaptor_layers(hidden_states, False, adaptor_layer_idx*3)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if self.config.multi_source and (self.config.multi_source_method == "merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "mid_fusion_merge_after_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention"):
            additional_cross_attn_weights = None
            additional_cross_attn_present_key_value = None

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            
            ## Modified by Raj Dabre. Start.
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            if self.config.multi_source and (self.config.multi_source_method == "merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "mid_fusion_merge_after_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention"): ## This if else is not needed but keeping it that way for cleaner flow of logic.
                cross_attn_past_key_value = past_key_value[-4:-2] if past_key_value is not None else None
                additional_cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
                hidden_states, cross_attn_weights, additional_cross_attn_weights, cross_attn_present_key_value, additional_cross_attn_present_key_value = self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    additional_key_value_states=additional_encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    additional_attention_mask=additional_encoder_attention_mask,
                    layer_head_mask=layer_head_mask, ## Should be none. Dont mess with this.
                    past_key_value=cross_attn_past_key_value,
                    additional_past_key_value=additional_cross_attn_past_key_value,
                    output_attentions=output_attentions, ## Should be false. Dont mess with this.
                )
                #print(hidden_states.size() if hidden_states is not None else 1, attention_mask.size() if attention_mask is not None else 1)
            else:
                cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
                hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )

            ## Modified by Raj Dabre. End.
            
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            
            ## Modified by Raj Dabre. Start.
            # add cross-attn to positions 3,4 of present_key_value tuple
            if self.config.multi_source and (self.config.multi_source_method == "merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "mid_fusion_merge_after_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention"):
                present_key_value = present_key_value + cross_attn_present_key_value + additional_cross_attn_present_key_value
            else:
                present_key_value = present_key_value + cross_attn_present_key_value ## Deal with the additional_cross_attn_present_key_value
            ## Modified by Raj Dabre. End.

        
        if adaptor_layers is not None and deep_adaptor_tuning: # Apply adaptor layer to current layer's output.
            hidden_states = adaptor_layers(hidden_states, False, adaptor_layer_idx*3+1)
        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if self.config.use_moe:
            hidden_states, moe_loss = self.moe(hidden_states)
        else:
            hidden_states = self.activation_fn(self.fc1(hidden_states))
            hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
            hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        
        if adaptor_layers is not None and deep_adaptor_tuning: # Apply adaptor layer to current layer's output.
            hidden_states = adaptor_layers(hidden_states, False, adaptor_layer_idx*3+2)

        if self.config.use_moe:
            outputs = ([hidden_states, moe_loss],)
        else:
            outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
            ## Modified by Raj Dabre. Start.
            if self.config.multi_source and (self.config.multi_source_method == "merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only"):
                outputs += (additional_cross_attn_weights,)
            ## Modified by Raj Dabre. End.
            
        if use_cache:
            outputs += (present_key_value,) ## Deal with the additional_cross_attn_present_key_value

        return outputs


# Copied from transformers.models.bart.modeling_bart.BartClassificationHead with Bart->MBart
class MBartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class MBartPreTrainedModel(PreTrainedModel):
    config_class = MBartConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


MBART_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.MBartConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

MBART_GENERATION_EXAMPLE = r"""
    Summarization example::

        >>> from transformers import MBartTokenizer, MBartForConditionalGeneration, MBartConfig

        >>> model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
        >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')

        >>> ARTICLE_TO_SUMMARIZE = "Meine Freunde sind cool, aber sie essen zu viel Kuchen."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    Mask filling example::

        >>> from transformers import MBartTokenizer, MBartForConditionalGeneration
        >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
        >>> # de_DE is the language symbol id <LID> for German
        >>> TXT = "</s> Meine Freunde sind <mask> nett aber sie essen zu viel Kuchen. </s> de_DE"

        >>> model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
        >>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
"""

MBART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.MBartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for translation and summarization training. By default, the model will create this tensor by
            shifting the :obj:`input_ids` to the right, following the paper.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.MBartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            MBart uses a specific language id token as the starting token for :obj:`decoder_input_ids` generation that
            varies according to source and target language, *e.g.* 25004 for `en_XX`, and 25003 for `de_DE`. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            For translation and summarization training, :obj:`decoder_input_ids` should be provided. If no
            :obj:`decoder_input_ids` is provided, the model will create this tensor by shifting the :obj:`input_ids` to
            the right for denoising pre-training following the paper.
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read :func:`modeling_mbart._prepare_decoder_inputs` and
            modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the heas is **masked**.

        decoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class MBartEncoder(MBartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`MBartEncoderLayer`.

    Args:
        config: MBartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        
        ## Modified by Raj Dabre. Start.
        if config.features_vocab_sizes is not None: ### Set up embedders for features
            self.features_embed_tokens = [nn.Embedding(feature_vocab_size, feature_embed_dim, self.padding_idx) for feature_vocab_size, feature_embed_dim in zip(config.features_vocab_sizes, config.features_embed_dims)]
            self.features_final_project = nn.Linear(embed_dim+sum(config.features_embed_dims), embed_dim, bias=False)
        else:
            self.features_embed_tokens = None
            self.features_final_project = None
        ## Modified by Raj Dabre. End.
        
        if config.no_positional_encoding_encoder:
            print("Using no positional encodings for encoder")
            self.embed_positions = 0
        else:
            if config.positional_encodings:
                print("Using positional encodings")
                self.embed_positions = MBartSinusoidalPositionalEmbedding(
                    config.max_position_embeddings,
                    embed_dim,
                    self.padding_idx,
                )
            else:
                print("Using positional embeddings")
                self.embed_positions = MBartLearnedPositionalEmbedding(
                    config.max_position_embeddings,
                    embed_dim,
                    self.padding_idx,
                )
        ## Modified by Raj Dabre. Start.
        if config.encoder_tying_config is not None: ## Create unique or shared layers as per sharing configuration.
            layer_idxs = config.encoder_tying_config.strip().split("-")
            unique_idxs = sorted(set(layer_idxs))
            self.unique_layers = nn.ModuleList([MBartEncoderLayer(config) for idx in unique_idxs])
            self.layers = [self.unique_layers[int(idx)-1] for idx in layer_idxs]
        else:
            self.layers = nn.ModuleList([MBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        if config.multi_source and (config.multi_source_method == "self_relevance" or config.multi_source_method == "self_relevance_and_merge_before_attention" or config.multi_source_method == "self_relevance_and_merge_after_attention" or config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only"): ## We should pass each input through a relevance mechanism which is sigmoid(Wx) where x is the representation of the input.
            self.self_relevance_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        ## Modified by Raj Dabre. End.
        if not config.no_embed_norm:
            self.layernorm_embedding = nn.LayerNorm(embed_dim)
        if config.multi_source and (config.multi_source_method == "mid_fusion_merge_before_attention" or config.multi_source_method == "mid_fusion_merge_after_attention" or config.multi_source_method == "bottleneck_mid_fusion_merge_before_attention" or config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention"):
            pass
        else:
            self.layer_norm = nn.LayerNorm(config.d_model)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        features_ids=None, ### A tuple or list of feature ids. Each should have the same dimension as input_ids
        additional_input_ids=None, ## Placeholder argument. Wont be used.
        additional_input_ids_mask=None, ## Placeholder argument. Wont be used.
        prompt_params=None, ## Prompts to be prepended to the encoder outputs.
        adaptor_layers=None, ## Adaptor layers to used in the encoder.
        deep_adaptor_tuning=False, ## Whether to use deep adaptor tuning or not.
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.MBartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            input_shape = inputs_embeds.size()[:-1]
            if prompt_params is not None:
                for prompt_param_idx in range(len(prompt_params)):
                    prompt_params[prompt_param_idx] = prompt_params[prompt_param_idx] * self.embed_scale
                    prompt_params[prompt_param_idx] = prompt_params[prompt_param_idx].repeat(input_shape[0], 1, 1)
                prompt_shape = prompt_params[0].size()[:-1]
                
            ## Modified by Raj Dabre. Start.
            if self.features_final_project is not None and self.features_embed_tokens is not None: ## Perform feature computation and concatenation and projection.
                features_embeds = [feature_embed_tokens(feature_id) for feature_embed_tokens, feature_input_id in zip(self.features_embed_tokens, features_ids)]
                all_embeds = [inputs_embeds] + features_embeds
                input_embeds = self.features_final_project(torch.cat(all_embeds, dim=-1))## Basic feature based model. Add relevance model here.
            ## Modified by Raj Dabre. End.
            
        if self.config.no_positional_encoding_encoder:
            embed_pos = self.embed_positions
            if prompt_params is not None:
                prompt_pos = self.embed_positions
        else:
            if prompt_params is not None:
                prompt_pos = self.embed_positions(prompt_shape, 0)
                embed_pos = self.embed_positions(input_shape, prompt_shape[1])
            else:
                embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        if prompt_params is not None:
            for prompt_param_idx in range(len(prompt_params)):
                prompt_params[prompt_param_idx] = prompt_params[prompt_param_idx] + prompt_pos

        if not self.config.no_embed_norm:
            hidden_states = self.layernorm_embedding(hidden_states)
            if prompt_params is not None:
                for prompt_param_idx in range(len(prompt_params)):
                    prompt_params[prompt_param_idx] = self.layernorm_embedding(prompt_params[prompt_param_idx])
        
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        
        if prompt_params is not None:
            for prompt_param_idx in range(len(prompt_params)):
                prompt_params[prompt_param_idx] = F.dropout(prompt_params[prompt_param_idx], p=self.dropout, training=self.training)
            hidden_states = torch.cat([prompt_params[0], hidden_states], dim=1)
                
        
        
        ## Modified by Raj Dabre. Start.
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype, wait_k=1 if self.config.wait_k!=-1 or self.config.unidirectional_encoder else -1) ## Raj: Just make the mask wait-k with a k=1 and we are good to go. We want to have a unidirectional encoder no matter what.
        ## Modified by Raj Dabre. End.

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        moe_losses = () if self.config.use_moe else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        adaptor_layers=adaptor_layers,
                        deep_adaptor_tuning=deep_adaptor_tuning,
                        adaptor_layer_idx=idx,
                    )
                
                if self.config.use_moe:
                    hidden_states, moe_loss = layer_outputs[0]
                    moe_losses += (moe_loss,)
                else:
                    hidden_states = layer_outputs[0]
                    
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            
            ### If prompts are used then we use the prompt embeddings instead of the updated representations of prompt embeddings when passed through a layer.
            if prompt_params is not None:
                hidden_states = torch.cat([prompt_params[idx+1], hidden_states[:,prompt_shape[1]:,:]], dim=1)

        ## Modified by Raj Dabre. Start.
        if self.config.multi_source and (self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_before_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only"):
            hidden_states = hidden_states*torch.sigmoid(self.self_relevance_layer(hidden_states)) # Do self relevance as usual.
        ## Modified by Raj Dabre. End.
        
        if adaptor_layers is not None and not deep_adaptor_tuning: ## Apply adaptor layer for final encoder layer.
            hidden_states = adaptor_layers(hidden_states, True)

        if self.config.multi_source and (self.config.multi_source_method == "mid_fusion_merge_before_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_before_attention" or self.config.multi_source_method == "mid_fusion_merge_after_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention"): # No layer norm because the fusion layers have not been processed yet.
            pass
        else:
            hidden_states = self.layer_norm(hidden_states)
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        
        
        
        
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions, moe_losses] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions, moe_losses=moe_losses
        )


class MBartDecoder(MBartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`MBartDecoderLayer`

    Args:
        config: MBartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        
        if config.no_positional_encoding_decoder:
            print("Using no positional encodings for decoder")
            self.embed_positions = 0
        else:
            if config.positional_encodings:
                print("Using positional encodings")
                self.embed_positions = MBartSinusoidalPositionalEmbedding(
                    config.max_position_embeddings,
                    config.d_model,
                    self.padding_idx,
                )
            else:
                print("Using positional embeddings")
                self.embed_positions = MBartLearnedPositionalEmbedding(
                    config.max_position_embeddings,
                    config.d_model,
                    self.padding_idx,
                )
        ## Modified by Raj Dabre. Start.
        if config.decoder_tying_config is not None: ## Create unique or shared layers as per sharing configuration.
            layer_idxs = config.decoder_tying_config.strip().split("-")
            unique_idxs = sorted(set(layer_idxs))
            self.unique_layers = nn.ModuleList([MBartDecoderLayer(config) for idx in unique_idxs])
            self.layers = [self.unique_layers[int(idx)-1] for idx in layer_idxs]
        else:
            self.layers = nn.ModuleList([MBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        ## Modified by Raj Dabre. End.
        if not config.no_embed_norm:
            self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length, prompting=False):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)
            if prompting:
                bsz, _, tgt_seq_len, src_seq_len = combined_attention_mask.size()
                combined_attention_mask = torch.cat([combined_attention_mask[:,:,0:1,:].expand(bsz, 1, past_key_values_length, src_seq_len), combined_attention_mask], dim=2)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_encoder_hidden_states=None,
        additional_encoder_attention_mask=None,
        curr_decode_length=-1,
        prompt_params=None, ## Prompts to be prepended to the decoder outputs.
        adaptor_layers=None, ## Adaptor layers to be used in the decoder.
        deep_adaptor_tuning=False, ## Whether to use deep adaptor tuning.
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.MBartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        
        if prompt_params is not None and (self.training or curr_decode_length == 1): ## During training past_key_values_length will always be 0 so it needs to be increased to get a proper causal decoder. Of course we need the input embeds to be augmented with the prompt info. During evaluation, this does not matter at all but we need the input embeds to be augmented with the prompt info during the first generation step.
            past_key_values_length += prompt_params[0].size()[1]
            batch_dims = inputs_embeds.size()
            for prompt_params_idx in range(len(prompt_params)):
                prompt_params[prompt_params_idx] = prompt_params[prompt_params_idx] * self.embed_scale
                prompt_params[prompt_params_idx] = prompt_params[prompt_params_idx].repeat(batch_dims[0], 1, 1) # Repeat the embeddings for each batch
            prompt_shape = prompt_params[0].size()[:-1]    
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length, prompting = prompt_params is not None
        ) ## Will be none if not training.
                
        ## Modified by Raj Dabre. Start.
        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1] +(prompt_shape[1] if prompt_params is not None and (self.training or curr_decode_length == 1) else 0), wait_k=self.config.wait_k, curr_decode_length=curr_decode_length) ## Raj: Just make the mask wait-k and we are good to go. We wont deal with wait-k and prompts at the moment since it gets a bit tricky. TODO: Make prompts and wait-k work together.
            if self.config.multi_source:
                if additional_encoder_hidden_states is not None and additional_encoder_attention_mask is not None:
                    additional_encoder_attention_mask = _expand_mask(additional_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1], wait_k=self.config.additional_source_wait_k, curr_decode_length=curr_decode_length) ## Raj: Just make the mask wait-k and we are good to go.
        # embed positions
        #print(encoder_attention_mask.size() if encoder_attention_mask is not None else 1, additional_encoder_attention_mask.size() if additional_encoder_attention_mask is not None else 1)
        ## Modified by Raj Dabre. End.
        if self.config.no_positional_encoding_decoder:
            positions = self.embed_positions
            if prompt_params is not None and (self.training or curr_decode_length == 1):
                prompt_positions = self.embed_positions
        else:
            if prompt_params is not None and (self.training or curr_decode_length == 1):
                prompt_positions = self.embed_positions(prompt_shape, 0)

            positions = self.embed_positions(inputs_embeds.size(), past_key_values_length) ## No matter what, the past key values length will be be properly updated.
            
        hidden_states = inputs_embeds + positions

        if prompt_params is not None and (self.training or curr_decode_length == 1):
            for prompt_params_idx in range(len(prompt_params)):
                prompt_params[prompt_params_idx] = prompt_params[prompt_params_idx] + prompt_positions

        if not self.config.no_embed_norm:
            hidden_states = self.layernorm_embedding(hidden_states)
            if prompt_params is not None and (self.training or curr_decode_length == 1):
                for prompt_params_idx in range(len(prompt_params)):
                    prompt_params[prompt_params_idx] = self.layernorm_embedding(prompt_params[prompt_params_idx])

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        if prompt_params is not None and (self.training or curr_decode_length == 1):
            for prompt_params_idx in range(len(prompt_params)):
                prompt_params[prompt_params_idx] = F.dropout(prompt_params[prompt_params_idx], p=self.dropout, training=self.training)
            hidden_states = torch.cat([prompt_params[0], hidden_states], dim=1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        moe_losses = () if self.config.use_moe else None
        ## Modified by Raj Dabre. Start.
        additional_all_cross_attentions = () if self.config.multi_source and (self.config.multi_source_method == "merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "mid_fusion_merge_after_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention") and output_attentions and additional_encoder_hidden_states is not None else None
        next_decoder_cache = () if use_cache else None
        if self.config.multi_source and (self.config.multi_source_method == "merge_before_attention" or self.config.multi_source_method == "self_relevance_and_merge_before_attention" or self.config.multi_source_method == "mid_fusion_merge_before_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_before_attention" ):
            encoder_hidden_states = torch.cat([encoder_hidden_states, additional_encoder_hidden_states], 1) ## Concatenate sequences blindly along the sequence axis. 
            encoder_attention_mask = torch.cat([encoder_attention_mask, additional_encoder_attention_mask], -1) ## Concatenate along the src_seq_len axis.
            #print(encoder_hidden_states.size(), encoder_attention_mask.size())
        
        ## Modified by Raj Dabre. End.
        
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    additional_encoder_hidden_states=additional_encoder_hidden_states,
                    additional_encoder_attention_mask=additional_encoder_attention_mask,
                    adaptor_layers=adaptor_layers,
                    deep_adaptor_tuning=deep_adaptor_tuning,
                    adaptor_layer_idx=idx,
                )
            if self.config.use_moe:
                hidden_states, moe_loss = layer_outputs[0]
                moe_losses += (moe_loss,)
            else:
                hidden_states = layer_outputs[0]
            
            # If prompts are used then we use the prompt embeddings instead of the updated representations of prompt embeddings when passed through a layer.
            if prompt_params is not None and (self.training or curr_decode_length == 1):
                hidden_states = torch.cat([prompt_params[idx+1], hidden_states[:, prompt_shape[1]:, :]], dim=1)

            ## Modified by Raj Dabre. Start.
            if use_cache:
                if self.config.multi_source and (self.config.multi_source_method == "merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "mid_fusion_merge_after_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention"):
                    next_decoder_cache += (layer_outputs[4 if output_attentions else 1],)
                else:
                    next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            ## Modified by Raj Dabre. End.
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
                    ## Modified by Raj Dabre. Start.
                    if self.config.multi_source and (self.config.multi_source_method == "merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "mid_fusion_merge_after_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention"):
                        additional_all_cross_attentions += (layer_outputs[3],)
                    ## Modified by Raj Dabre. End.

        if adaptor_layers is not None and not deep_adaptor_tuning: ## Apply adaptor layer for final decoder output only.
            hidden_states = adaptor_layers(hidden_states, False)
            
        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        ## Modified by Raj Dabre. Start.
        if not return_dict:
            if self.config.multi_source and (self.config.multi_source_method == "merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "mid_fusion_merge_after_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention"):
                return tuple(
                    v
                    for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions, additional_all_cross_attentions, moe_losses]
                    if v is not None
                )
            else:
                return tuple(
                    v
                    for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions, moe_losses]
                    if v is not None
                )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            additional_cross_attentions=additional_all_cross_attentions, 
            moe_losses = moe_losses,
        )
        ## Modified by Raj Dabre. End.


@add_start_docstrings(
    "The bare MBART Model outputting raw hidden-states without any specific head on top.",
    MBART_START_DOCSTRING,
)
class MBartModel(MBartPreTrainedModel):
    def __init__(self, config: MBartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MBartEncoder(config, self.shared)
        self.decoder = MBartDecoder(config, self.shared)
        
        ## Modified by Raj Dabre. Start.
        if self.config.multi_source and config.multi_source_method == "additional_source_attention":
            self.context_attention = MBartDecoderLayer(config)
            self.context_norm = nn.LayerNorm(config.d_model)

        if self.config.multi_source and (config.multi_source_method == "mid_fusion_merge_before_attention" or config.multi_source_method == "bottleneck_mid_fusion_merge_before_attention" or config.multi_source_method == "mid_fusion_merge_after_attention" or config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention"):
            self.mid_fusion_layers = nn.ModuleList([MBartEncoderLayer(config) for _ in range(config.mid_fusion_layers)])
            self.mid_fusion_norm = nn.LayerNorm(config.d_model)
            if config.multi_source_method == "bottleneck_mid_fusion_merge_before_attention" or config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention":
                bottleneck_params = torch.zeros(1, config.bottleneck_mid_fusion_tokens, config.d_model)
                bottleneck_params.normal_(mean=0.0, std=config.init_std)
                self.bottleneck_params = torch.nn.Parameter(bottleneck_params)
        ## Modified by Raj Dabre. End.
        
        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/mbart-large-cc25",
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_input_ids=None,
        additional_input_ids_mask=None,
        additional_encoder_outputs=None,
        context_encoder_representations=None,
        curr_decode_length=-1,
        prompt_params=None,
        adaptor_layers=None,
        deep_adaptor_tuning=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # different to other models, MBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                prompt_params=prompt_params[0] if prompt_params is not None else None,
                adaptor_layers=adaptor_layers,
                deep_adaptor_tuning=deep_adaptor_tuning,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        ## Modified by Raj Dabre. Start.
        if self.config.multi_source:
            if additional_encoder_outputs is None:
                main_source_wait_k = self.config.wait_k
                self.config.wait_k = self.config.additional_source_wait_k
                additional_encoder_outputs = self.encoder(
                    input_ids=additional_input_ids,
                    attention_mask=additional_input_ids_mask,
                    head_mask=head_mask, ## Should be None. Dont mess with this.
                    inputs_embeds=inputs_embeds, ## Should be None. Dont mess with this.
                    output_attentions=output_attentions, ## Should be False. Dont mess with this.
                    output_hidden_states=output_hidden_states, ## Should be False. Dont mess with this.
                    return_dict=return_dict,
                )
                if self.config.use_moe: ## Add the additional encoder MOE losses to the main encoder.
                    encoder_outputs[3] = encoder_outputs[3] + additional_encoder_outputs[3]

                self.config.wait_k = main_source_wait_k
            # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
            elif return_dict and not isinstance(additional_encoder_outputs, BaseModelOutput):
                additional_encoder_outputs = BaseModelOutput(
                    last_hidden_state=additional_encoder_outputs[0],
                    hidden_states=additional_encoder_outputs[1] if len(additional_encoder_outputs) > 1 else None,
                    attentions=additional_encoder_outputs[2] if len(additional_encoder_outputs) > 2 else None,
                ) ## Figure out a way to return this
        else:
            additional_encoder_outputs = [None]
        
        if self.config.multi_source and (self.config.multi_source_method == "mid_fusion_merge_before_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_before_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention" or self.config.multi_source_method == "mid_fusion_merge_after_attention"): 
            # Concatenate the encoder and additional encoder outputs or concatenate the bottleneck params with the encoder and additional encoder outputs
            # Create encoder layers to deal with further processing
            # Do processing, deal with MOEs etc, update the hidden states after splitting at each stage etc
            # We will need a new hyperparam to tell us the number of additional layers. Will have to deal with recurrent stacking etc. additional layers and current layers sum to actual total layers.
            # Disable layer norm in the main encoder code when this type of fusion is done. 
            if context_encoder_representations is None:
                hidden_states = encoder_outputs[0]
                additional_hidden_states = additional_encoder_outputs[0]
                encoder_input_length = hidden_states.size()[1]
                additional_encoder_input_length = additional_hidden_states.size()[1]
                encoder_self_attention_mask = _expand_mask(attention_mask, hidden_states.dtype, wait_k=self.config.wait_k)
                additional_encoder_self_attention_mask = _expand_mask(additional_input_ids_mask, additional_hidden_states.dtype, wait_k=self.config.additional_source_wait_k)
                if self.config.multi_source_method == "mid_fusion_merge_before_attention" or self.config.multi_source_method == "mid_fusion_merge_after_attention":
                    # Concatenate the encoder and additional encoder outputs
                    # We have to deal with creation of attention masks for encoder to itself, additional encoder to itself and then cross between these two.
                    encoder_to_additional_encoder_self_attention_mask = _expand_mask(additional_input_ids_mask, additional_hidden_states.dtype, tgt_len=encoder_input_length, wait_k=self.config.wait_k)
                    additional_encoder_to_encoder_self_attention_mask = _expand_mask(attention_mask, hidden_states.dtype, tgt_len=additional_encoder_input_length, wait_k=self.config.additional_source_wait_k)
                    combined_mask_a = torch.cat([encoder_self_attention_mask, encoder_to_additional_encoder_self_attention_mask], dim=3)
                    combined_mask_b = torch.cat([additional_encoder_to_encoder_self_attention_mask, additional_encoder_self_attention_mask], dim=3)
                    combined_mask = torch.cat((combined_mask_a, combined_mask_b), dim=2)
                    combined_encoder_outputs = torch.cat((hidden_states, additional_hidden_states), dim=1)
                    for idx, fusion_layer in enumerate(self.mid_fusion_layers):
                        if output_hidden_states:
                            encoder_outputs[1] = encoder_outputs[1] + (hidden_states,)
                            additional_encoder_outputs[1] = additional_encoder_outputs[1] + (additional_hidden_states,)
                        layer_outputs = fusion_layer(
                            combined_encoder_outputs,
                            combined_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                            adaptor_layers=adaptor_layers,
                            deep_adaptor_tuning=deep_adaptor_tuning,
                            adaptor_layer_idx=idx+self.config.encoder_layers,
                        )

                        if self.config.use_moe:
                            hidden_states, moe_loss = layer_outputs[0]
                            encoder_outputs[3] = encoder_outputs[3] + moe_loss
                        else:
                            hidden_states = layer_outputs[0]
                        
                        combined_encoder_outputs = hidden_states
                        # Split hidden states and update the hidden states
                        hidden_states, additional_hidden_states = torch.split(hidden_states, (encoder_input_length, additional_encoder_input_length), dim=1)
                    
                        if output_attentions:
                            current_attentions = layer_outputs[1]
                            # Split the attentions and update the attentions
                            current_attentions, additional_current_attentions = torch.split(current_attentions, (encoder_input_length, additional_encoder_input_length), dim=2) ## This needs to be fixed. Since the number of columns will be (encoder_input_length+additional_encoder_input_length). But this may or may not be important.
                            encoder_outputs[2] = encoder_outputs[2] + (current_attentions,)
                            additional_encoder_outputs[2] = additional_encoder_outputs[2] + (additional_current_attentions,)
                elif self.config.multi_source_method == "bottleneck_mid_fusion_merge_before_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention":
                    batch_size = hidden_states.size()[0]
                    # Expand the bottleneck params to batch size
                    bottleneck_params = self.bottleneck_params.expand(batch_size, -1, -1)
                    # Concatenate the bottleneck params with the encoder and additional encoder outputs individually
                    combined_hidden_states = torch.cat((bottleneck_params, hidden_states), dim=1)
                    combined_additional_hidden_states = torch.cat((bottleneck_params, additional_hidden_states), dim=1)
                    # Create a ones mask of shape (batch_size, 1, encoder_input_length, bottleneck_mid_fusion_tokens)
                    ones_mask = torch.ones(batch_size, 1, encoder_input_length, self.config.bottleneck_mid_fusion_tokens).to(hidden_states.device)
                    additional_ones_mask = torch.ones(batch_size, 1, additional_encoder_input_length, self.config.bottleneck_mid_fusion_tokens).to(hidden_states.device)
                    
                    # Expand the masks to accommodate the bottleneck params. We replicate the first row bottleneck_mid_fusion_tokens number of times. Its ok since bottleneck params can attend to itself and should attend to the first token. 
                    encoder_self_attention_mask = torch.cat((ones_mask, encoder_self_attention_mask), dim=3) 
                    encoder_self_attention_mask = torch.cat([encoder_self_attention_mask[:,:,0:1,:].expand(batch_size, 1, self.config.bottleneck_mid_fusion_tokens, self.config.bottleneck_mid_fusion_tokens+encoder_input_length), encoder_self_attention_mask], dim=2)
                    additional_encoder_self_attention_mask = torch.cat((additional_ones_mask, additional_encoder_self_attention_mask), dim=3)
                    additional_encoder_self_attention_mask = torch.cat([additional_encoder_self_attention_mask[:,:,0:1,:].expand(batch_size, 1, self.config.bottleneck_mid_fusion_tokens, self.config.bottleneck_mid_fusion_tokens+additional_encoder_input_length), additional_encoder_self_attention_mask], dim=2)

                    for idx, fusion_layer in enumerate(self.mid_fusion_layers):
                        if output_hidden_states:
                            encoder_outputs[1] = encoder_outputs[1] + (hidden_states,)
                            additional_encoder_outputs[1] = additional_encoder_outputs[1] + (additional_hidden_states,)
                        layer_outputs = fusion_layer(
                            combined_hidden_states,
                            encoder_self_attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                            adaptor_layers=adaptor_layers,
                            deep_adaptor_tuning=deep_adaptor_tuning,
                            adaptor_layer_idx=idx+self.config.encoder_layers,
                        )

                        additional_layer_outputs = fusion_layer(
                            combined_additional_hidden_states,
                            additional_encoder_self_attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                            adaptor_layers=adaptor_layers,
                            deep_adaptor_tuning=deep_adaptor_tuning,
                            adaptor_layer_idx=idx+self.config.encoder_layers,
                        )

                        if self.config.use_moe:
                            combined_hidden_states, moe_loss = layer_outputs[0]
                            combined_additional_hidden_states, additional_moe_loss = additional_layer_outputs[0]
                            encoder_outputs[3] = encoder_outputs[3] + moe_loss + additional_moe_loss
                        else:
                            combined_hidden_states = layer_outputs[0]
                            combined_additional_hidden_states = additional_layer_outputs[0]
                        
                        # Split hidden states and additionl hidden states bu discarding the bottleneck parts
                        bottleneck_params, hidden_states  = torch.split(combined_hidden_states, (self.config.bottleneck_mid_fusion_tokens, encoder_input_length), dim=1)
                        additional_bottleneck_params, additional_hidden_states  = torch.split(combined_additional_hidden_states, (self.config.bottleneck_mid_fusion_tokens, additional_encoder_input_length), dim=1)

                        # Average the bottleneck params
                        bottleneck_params = (bottleneck_params + additional_bottleneck_params)/2

                        # Concatenate the bottleneck params with the encoder and additional encoder outputs individually
                        combined_hidden_states = torch.cat((bottleneck_params, hidden_states), dim=1)
                        combined_additional_hidden_states = torch.cat((bottleneck_params, additional_hidden_states), dim=1)
                        
                        if output_attentions:
                            current_attentions = layer_outputs[1]
                            additional_current_attentions = additional_layer_outputs[1]
                            # Split the attentions and update the attentions
                            ## This needs to be fixed. Since the number of columns will be (self.config.bottleneck_mid_fusion_tokens+encoder_input_length) and (self.config.bottleneck_mid_fusion_tokens+additional_encoder_input_length). But this may or may not be important.
                            current_attentions = torch.split(current_attentions, (self.config.bottleneck_mid_fusion_tokens, encoder_input_length), dim=2)[1]
                            additional_current_attentions = torch.split(additional_current_attentions, (self.config.bottleneck_mid_fusion_tokens, additional_encoder_input_length), dim=2)[1]
                            encoder_outputs[2] = encoder_outputs[2] + (current_attentions,)
                            additional_encoder_outputs[2] = additional_encoder_outputs[2] + (additional_current_attentions,)
                
                # Apply the layer normalization
                hidden_states = self.mid_fusion_norm(hidden_states)
                additional_hidden_states = self.mid_fusion_norm(additional_hidden_states)

                # Update the hidden states
                encoder_outputs["last_hidden_state"] = hidden_states
                additional_encoder_outputs["last_hidden_state"] = additional_hidden_states
                context_encoder_representations = torch.cat((hidden_states, additional_hidden_states), dim=1) ## We use this as a placeholder to prevent any additional computations :)
            

        if self.config.multi_source and self.config.multi_source_method == "additional_source_attention": ## We do a "cross attention" between the sentence and its context. For now this will be recomputed for each decoding time step.
            if context_encoder_representations is None: 
                encoder_input_length = encoder_outputs[0].size()[1]
                additional_encoder_input_length = additional_encoder_outputs[0].size()[1]
                encoder_self_attention_mask = _expand_mask(attention_mask, encoder_outputs[0].dtype, wait_k=self.config.additional_source_wait_k)
                encoder_encoder_cross_attention_mask = _expand_mask(additional_input_ids_mask, encoder_outputs[0].dtype, tgt_len=encoder_input_length, wait_k=self.config.additional_source_wait_k)

                context_encoder_representations = self.context_attention(encoder_outputs[0],
                        attention_mask=encoder_self_attention_mask,
                        encoder_hidden_states=additional_encoder_outputs[0],
                        encoder_attention_mask=encoder_encoder_cross_attention_mask,
                        layer_head_mask=None,
                        encoder_layer_head_mask=None,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                        additional_encoder_hidden_states=None,
                        additional_encoder_attention_mask=None,)
                context_encoder_representations[0] = self.context_norm(context_encoder_representations[0]) 
                #print(type(encoder_outputs), type(context_encoder_representations))
                encoder_outputs["last_hidden_state"] = context_encoder_representations[0]
                context_encoder_representations = context_encoder_representations[0]

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states or self.config.multilayer_softmaxing is not None, ## In case of multilayer softmaxing we need the hidden states ONLY FROM THE DECODER.
            return_dict=return_dict,
            additional_encoder_hidden_states=additional_encoder_outputs[0],
            additional_encoder_attention_mask=additional_input_ids_mask,
            curr_decode_length=curr_decode_length,
            prompt_params=prompt_params[1] if prompt_params is not None else None,
            adaptor_layers=adaptor_layers,
            deep_adaptor_tuning=deep_adaptor_tuning,
        )

        if prompt_params is not None and (self.training or curr_decode_length == 1):
            decoder_outputs.last_hidden_state = decoder_outputs.last_hidden_state[:,prompt_params[1][0].size()[1]:,:]
        
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            additional_encoder_last_hidden_state=additional_encoder_outputs.last_hidden_state if self.config.multi_source else None,
            additional_encoder_hidden_states=additional_encoder_outputs.hidden_states if self.config.multi_source else None,
            additional_encoder_attentions=additional_encoder_outputs.attentions if self.config.multi_source else None,
            additional_cross_attentions=decoder_outputs.additional_cross_attentions if self.config.multi_source and (self.config.multi_source_method == "merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "mid_fusion_merge_after_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention") else (),
            context_encoder_representations = context_encoder_representations if self.config.multi_source and (self.config.multi_source_method == "additional_source_attention" or self.config.multi_source_method == "mid_fusion_merge_before_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_before_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention" or self.config.multi_source_method == "mid_fusion_merge_after_attention") else None, ## Find a way to return all contents of context_encoder_representations in the future.
            encoder_moe_losses = encoder_outputs.moe_losses, 
            decoder_moe_losses = decoder_outputs.moe_losses, 
        )
        ## Modified by Raj Dabre. End.

## Modified by Raj Dabre. Start.

class GradientReversalFunction(Function): ## Glory be to the gradients in reverse. AMEN!
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Prompts(nn.Module):
    """Custom Pytorch model for creating continuous prompts.
    """
    def __init__(self, num_prompts, d_model, init_std):
        
        super().__init__()
        # initialize weights with random numbers
        prompt_params = torch.zeros(1, num_prompts, d_model)
        prompt_params.normal_(mean=0.0, std=init_std)
        self.prompt_params = torch.nn.Parameter(prompt_params)
        ffn1 = torch.zeros(d_model, d_model*4)
        ffn1.normal_(mean=0.0, std=init_std)
        self.ffn1 = torch.nn.Parameter(ffn1)
        self.activation = torch.nn.GELU()
        ffn2 = torch.zeros(d_model*4, d_model)
        ffn2.normal_(mean=0.0, std=init_std)
        self.ffn2 = torch.nn.Parameter(ffn2)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, dummy_arg):
        return self.prompt_params + torch.matmul(self.activation(torch.matmul(self.layer_norm(self.prompt_params), self.ffn1)), self.ffn2)
    
class EncoderDecoderPrompts(nn.Module):
    """Custom Pytorch model for creating continuous prompts.
    """
    def __init__(self, num_prompts, encoder_layers, decoder_layers, d_model, init_std):
        
        super().__init__()
        # initialize weights with random numbers
        self.decoder_prompts = torch.nn.ModuleList([Prompts(num_prompts, d_model, init_std) for _ in range(encoder_layers+1)])
        self.encoder_prompts = torch.nn.ModuleList([Prompts(num_prompts, d_model, init_std) for _ in range(decoder_layers+1)])
        print("Number of additional parameters during training are:", (encoder_layers+1)*(d_model*d_model*4*2+ num_prompts*d_model)+(decoder_layers+1)*(d_model*d_model*4*2+ num_prompts*d_model))
        print("Number of additional parameters during evaluation are:", (encoder_layers+1)*(num_prompts*d_model)+(decoder_layers+1)*(num_prompts*d_model))
        self.num_prompts = num_prompts
        self.d_model = d_model
        
    def forward(self, dummy_arg):
        return [encoder_prompt(dummy_arg) for encoder_prompt in self.encoder_prompts], [decoder_prompt(dummy_arg) for decoder_prompt in self.decoder_prompts]


class Adaptor(nn.Module):
    """Custom Pytorch model for adaptor FFNs. We will pass these to the model and optimize and save them separately.
    """
    def __init__(self, d_model, hidden, init_std=0.02, hypercomplex=False, hypercomplex_n=2):
        
        super().__init__()
        # initialize weights with random numbers
        if hypercomplex:
            self.ffn1_a = torch.nn.ModuleList()
            self.ffn2_a = torch.nn.ModuleList()
            self.ffn1_b = torch.nn.ModuleList()
            self.ffn2_b = torch.nn.ModuleList()
            for _ in range(hypercomplex_n):
                ffn1_a = torch.zeros(hypercomplex_n, hypercomplex_n)
                ffn1_a.normal_(mean=0.0, std=init_std)
                self.ffn1_a.append(torch.nn.Parameter(ffn1_a))
                ffn2_a = torch.zeros(hypercomplex_n, hypercomplex_n)
                ffn2_a.normal_(mean=0.0, std=init_std)
                self.ffn2_a.append(torch.nn.Parameter(ffn2_a))
                ffn1_b = torch.zeros(d_model/hypercomplex_n, hidden/hypercomplex_n)
                ffn1_b.normal_(mean=0.0, std=init_std)
                self.ffn1_b.append(torch.nn.Parameter(ffn1_b))
                ffn2_b = torch.zeros(hidden/hypercomplex_n, d_model/hypercomplex_n)
                ffn2_b.normal_(mean=0.0, std=init_std)
                self.ffn2_b.append(torch.nn.Parameter(ffn2_b))
            self.ffn1 = torch.sum(torch.stack([torch.kron(self.ffn1_a[i], self.ffn1_b[i]) for i in range(hypercomplex_n)]), 0)
            self.ffn2 = torch.sum(torch.stack([torch.kron(self.ffn2_a[i], self.ffn2_b[i]) for i in range(hypercomplex_n)]), 0)    
        else:
            ffn1 = torch.zeros(d_model, hidden)
            ffn1.normal_(mean=0.0, std=init_std)
            self.ffn1 = torch.nn.Parameter(ffn1)
            ffn2 = torch.zeros(hidden, d_model)
            ffn2.normal_(mean=0.0, std=init_std)
            self.ffn2 = torch.nn.Parameter(ffn2)
        
        self.activation = torch.nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, input):
        return input + torch.matmul(self.activation(torch.matmul(self.layer_norm(input), self.ffn1)), self.ffn2) # Don't forget the residual connection
   

class EncoderDecoderAdaptors(nn.Module):
    """Custom Pytorch model for creating encoder-decoder adaptors. These adaptors will only be applied to the top encoder and decoder layer.
    """
    def __init__(self, d_model, hidden, init_std=0.02, hypercomplex=False, hypercomplex_n=2):
        
        super().__init__()
        # initialize weights with random numbers
        self.encoder_adaptor = Adaptor(d_model, hidden, init_std=init_std, hypercomplex=hypercomplex, hypercomplex_n=hypercomplex_n)
        self.decoder_adaptor = Adaptor(d_model, hidden, init_std=init_std, hypercomplex=hypercomplex, hypercomplex_n=hypercomplex_n)
        if hypercomplex:
            print("Hypercomplex adaptors will be used.")
            print("Number of additional parameters during training are:", (d_model*hidden*2*2)/hypercomplex_n + hypercomplex_n**3)
        else:
            print("Number of additional parameters during training are:", (d_model*hidden*2*2))
        
    def forward(self, input, is_encoder):
        if is_encoder:
            return self.encoder_adaptor(input)
        else:
            return self.decoder_adaptor(input)

class DeepEncoderDecoderAdaptors(nn.Module):
    """Custom Pytorch model for creating encoder-decoder adaptors. These adaptors will be applied after each layer.
    The adaptors should be lightweight with small hidden params.
    """
    def __init__(self, d_model, hidden, encoder_layers, decoder_layers, init_std=0.02, hypercomplex=False, hypercomplex_n=2):
        
        super().__init__()
        # initialize weights with random numbers
        self.encoder_adaptors = torch.nn.ModuleList([Adaptor(d_model, hidden, init_std=init_std, hypercomplex=hypercomplex, hypercomplex_n=hypercomplex_n) for _ in range(encoder_layers*2)])
        self.decoder_adaptors = torch.nn.ModuleList([Adaptor(d_model, hidden, init_std=init_std, hypercomplex=hypercomplex, hypercomplex_n=hypercomplex_n) for _ in range(decoder_layers*3)])
        if hypercomplex:
            print("Hypercomplex adaptors will be used.")
            print("Number of additional parameters during training are:", ((d_model*hidden*2)/hypercomplex_n + hypercomplex_n**3)*(encoder_layers*2+decoder_layers*3))
        else:
            print("Number of additional parameters during training are:", (d_model*hidden*2)*(encoder_layers*2+decoder_layers*3))
        
    def forward(self, input, is_encoder, layer_idx):
        if is_encoder:
            return self.encoder_adaptors[layer_idx](input)
        else:
            return self.decoder_adaptors[layer_idx](input)



@add_start_docstrings(
    "The MBART Model with a language modeling head. Can be used for summarization.", MBART_START_DOCSTRING
)
class MBartForConditionalGeneration(MBartPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
        r"prompt_params",
        r"adaptor_layers",
    ]

    def __init__(self, config: MBartConfig):
        super().__init__(config)
        self.model = MBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        
        if config.multilayer_softmaxing is not None:
            config.multilayer_softmaxing = [int(layer_id) for layer_id in config.multilayer_softmaxing.split(",")]
        
        self.init_weights()
        if config.temperature_calibration:
            assert config.softmax_temperature == 1.0
            print("Temperature calibration will be done.")
            self.register_parameter("softmax_temperature", torch.ones(1))
            print("Initial temperature is: ", self.softmax_temperature)
            
        if config.num_domains_for_domain_classifier > 1:
            print("Domain classifier will be used.")
            self.domain_classifer_head = nn.Linear(config.d_model, config.num_domains_for_domain_classifier, bias=False)
            if config.gradient_reversal_for_domain_classifier:
                self.gradient_reversal_layer = GradientReversal()

        if config.prompt_tuning:
            print("Prompt tuning will be done.")
            self.prompt_params = EncoderDecoderPrompts(config.num_prompts, config.encoder_layers, config.decoder_layers, config.d_model, config.init_std)
        
        if config.adaptor_tuning:
            if config.deep_adaptor_tuning:
                print("Deep adaptor tuning will be done.")
                self.adaptor_layers = DeepEncoderDecoderAdaptors(config.d_model, config.adaptor_hidden_size, config.encoder_layers, config.decoder_layers, config.init_std, config.hypercomplex, config.hypercomplex_n)
            else:
                print("Shallow adaptor tuning will be done.")
                self.adaptor_layers = EncoderDecoderAdaptors(config.d_model, config.adaptor_hidden_size, config.init_std, config.hypercomplex, config.hypercomplex_n)
        
        if config.softmax_bias_tuning:
            print("Softmax bias tuning will be done. Replacing the final logits bias with a learnable parameter.")
            self.final_logits_bias = nn.Parameter(torch.zeros(self.model.shared.num_embeddings).normal_(mean=0.0, std=config.init_std))

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def initialize_prompt_params_with_random_embeddings(self):
        """ Using random prompts as initial params is bad. Apparently its better to use random pretrained embeddings from the model.
        """
        print("Initializing prompt params with random embedding weights.")
        embeds = self.model.shared.weight.detach().clone().requires_grad_(True)
        num_embeds = embeds.size()[0]
        num_prompts = self.config.num_prompts
        with torch.no_grad():
            for i in range(len(self.prompt_params.encoder_prompts)):
                for prompt_id in range(num_prompts):
                    self.prompt_params.encoder_prompts[i].prompt_params[0, prompt_id, :] = embeds[random.randint(0, num_embeds-1)] ##  initialize with existing embeddings
            for i in range(len(self.prompt_params.decoder_prompts)):
                for prompt_id in range(num_prompts):
                    self.prompt_params.decoder_prompts[i].prompt_params[0, prompt_id, :] = embeds[random.randint(0, num_embeds-1)] ##  initialize with existing embeddings

    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MBART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_input_ids=None,
        additional_input_ids_mask=None,
        additional_encoder_outputs=None,
        additional_past_key_values=None,
        curr_decode_length=-1,
        context_encoder_representations=None,
        label_mask=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        
        
        if self.config.multi_source_method == "average_softmaxes":
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                additional_input_ids=None,
                additional_input_ids_mask=None,
                additional_encoder_outputs=None,
                curr_decode_length=curr_decode_length,
            )
            lm_logits = (self.lm_head(outputs[0]) + self.final_logits_bias)/self.config.softmax_temperature ## Divide the logits by a temperature to get a smoothed softmax.
            if self.config.temperature_calibration:
                lm_logits = lm_logits/self.softmax_temperature ## The softmax_temperature config param should be 1.0
            additional_outputs = self.model(
                additional_input_ids,
                attention_mask=additional_input_ids_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=additional_encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                past_key_values=additional_past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                additional_input_ids=None,
                additional_input_ids_mask=None,
                additional_encoder_outputs=None,
                curr_decode_length=curr_decode_length,
            )
            additional_source_lm_logits = (self.lm_head(additional_outputs[0]) + self.final_logits_bias)/self.config.softmax_temperature ## Divide the logits by a temperature to get a smoothed softmax.
            if self.config.temperature_calibration:
                additional_source_lm_logits = additional_source_lm_logits/self.softmax_temperature ## The softmax_temperature config param should be 1.0
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                additional_input_ids=additional_input_ids,
                additional_input_ids_mask=additional_input_ids_mask,
                additional_encoder_outputs=additional_encoder_outputs,
                curr_decode_length=curr_decode_length,
                context_encoder_representations=context_encoder_representations,
                prompt_params=self.prompt_params(0) if self.config.prompt_tuning else None,
                adaptor_layers=self.adaptor_layers if self.config.adaptor_tuning else None,
                deep_adaptor_tuning=self.config.deep_adaptor_tuning,
            )
            lm_logits = (self.lm_head(outputs[0]) + self.final_logits_bias)/self.config.softmax_temperature ## Divide the logits by a temperature to get a smoothed softmax.
            if self.config.temperature_calibration:
                lm_logits = lm_logits/self.softmax_temperature
        
        additional_lm_logits = []
        if self.config.multilayer_softmaxing is not None:
            for layer_id in self.config.multilayer_softmaxing: ## We count the embedding layer too. Who knows what may happen? However we wont do anything for the final layer as its already dealt with.
                lm_representation = outputs.decoder_hidden_states[layer_id]
                additional_lm_logits.append((self.lm_head(lm_representation) + self.final_logits_bias)/self.config.softmax_temperature) ## The additional logits will be collected here and then returned to my main code. Divide the logits by a temperature to get a smoothed softmax.
                if self.config.temperature_calibration:
                    additional_lm_logits[-1] = additional_lm_logits[-1]/self.softmax_temperature ## The softmax_temperature config param should be 1.0
        
        if self.config.num_domains_for_domain_classifier > 1: ## Pool the output layer representations by taking a mean and then generate logits for them.
            dom_pooled_outputs = outputs[0].masked_fill(label_mask, 0.0).mean(dim=1)
            if self.config.gradient_reversal_for_domain_classifier: ## If we want to do gradient reversal then thats going ot be done here.
                dom_pooled_outputs = self.gradient_reversal_layer(dom_pooled_outputs) 
            domain_classifier_logits = self.domain_classifer_head(dom_pooled_outputs)
            
            
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            additional_lm_logits=additional_lm_logits, 
            additional_encoder_last_hidden_state=outputs.additional_encoder_last_hidden_state if self.config.multi_source else None,
            additional_encoder_hidden_states=outputs.additional_encoder_hidden_states if self.config.multi_source else None,
            additional_encoder_attentions=outputs.additional_encoder_attentions if self.config.multi_source else None,
            additional_cross_attentions=outputs.additional_cross_attentions if self.config.multi_source and (self.config.multi_source_method == "merge_after_attention" or self.config.multi_source_method == "self_relevance_and_merge_after_attention" or self.config.multi_source_method == "merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "self_relevance_and_merge_after_attention_with_context_relevance_only" or self.config.multi_source_method == "average_softmaxes" or self.config.multi_source_method == "mid_fusion_merge_after_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention") else (),
            additional_past_key_values=additional_outputs.past_key_values if self.config.multi_source and (self.config.multi_source_method == "average_softmaxes") else None,
            additional_source_lm_logits=additional_source_lm_logits if self.config.multi_source and (self.config.multi_source_method == "average_softmaxes") else None,
            context_encoder_representations = outputs.context_encoder_representations if self.config.multi_source and (self.config.multi_source_method == "additional_source_attention" or self.config.multi_source_method == "mid_fusion_merge_before_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_before_attention" or self.config.multi_source_method == "bottleneck_mid_fusion_merge_after_attention" or self.config.multi_source_method == "mid_fusion_merge_after_attention") else None,
            softmax_temperature = self.softmax_temperature if self.config.temperature_calibration else None,
            domain_classifier_logits = domain_classifier_logits if self.config.num_domains_for_domain_classifier > 1 else None,
            encoder_moe_losses = outputs.encoder_moe_losses, 
            decoder_moe_losses = outputs.decoder_moe_losses, 
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "additional_input_ids": None,  # additional_encoder_outputs is defined. additional_input_ids not needed
            "additional_input_ids_mask": kwargs["additional_input_ids_mask"] if self.config.multi_source else None, ## This will contain the additional encoder outputs. 
            "additional_encoder_outputs": kwargs["additional_encoder_outputs"] if self.config.multi_source else None, ## This will contain the additional encoder outputs. 
            "additional_past_key_values": kwargs["additional_past"] if self.config.multi_source_method == "average_softmaxes" and "additional_past" in kwargs else None, ## This is for the past of the additional source when averaging softmaxes. 
            "context_encoder_representations": kwargs["context_encoder_representations"] if self.config.multi_source else None, ##  A bit sloppy and should be controlled by an additional condition looking at the value of multi_source type.
        }

## Modified by Raj Dabre. End.

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@add_start_docstrings(
    """
    MBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    MBART_START_DOCSTRING,
)
class MBartForSequenceClassification(MBartPreTrainedModel):
    def __init__(self, config: MBartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MBartModel(config)
        self.classification_head = MBartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/mbart-large-cc25",
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bart.modeling_bart.BartForSequenceClassification.forward
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@add_start_docstrings(
    """
    MBART Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MBART_START_DOCSTRING,
)
class MBartForQuestionAnswering(MBartPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = MBartModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.model._init_weights(self.qa_outputs)

    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/mbart-large-cc25",
        output_type=Seq2SeqQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bart.modeling_bart.BartForQuestionAnswering.forward
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        start_positions=None,
        end_positions=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


# Copied from transformers.models.bart.modeling_bart.BartDecoderWrapper with Bart->MBart
class MBartDecoderWrapper(MBartPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the :class:`~transformers.EncoderDecoderModel` framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = MBartDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


# Copied from transformers.models.bart.modeling_bart.BartForCausalLM with Bart->MBart
class MBartForCausalLM(MBartPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        self.model = MBartDecoderWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.MBartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last ``decoder_input_ids``
                (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
                instead of all ``decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
                config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are
                ignored (masked), the loss is only computed for the tokens with labels in ``[0, ...,
                config.vocab_size]``.
            use_cache (:obj:`bool`, `optional`):
                If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
                decoding (see :obj:`past_key_values`).

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.

        Returns:

        Example::

            >>> from transformers import MBartTokenizer, MBartForCausalLM

            >>> tokenizer = MBartTokenizer.from_pretrained('facebook/bart-large')
            >>> model = MBartForCausalLM.from_pretrained('facebook/bart-large', add_cross_attention=False)
            >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> last_hidden_states = outputs.last_hidden_state
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            encoder_head_mask=encoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past:
            input_ids = input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past