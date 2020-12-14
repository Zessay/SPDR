# coding=utf-8
# @Author: 莫冉
# @Date: 2020-06-05
"""
用于Rewrite Transformer Encoder-Decoder中的Decoder部分
"""
from typing import Tuple, Dict, Optional
import copy
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.nn import util

from resolution.common.modules.decoders import DecoderNet
from resolution.common.modules.encoders import (subsequent_mask, attention, PositionalEncoding, PositionwiseFeedForward)


@DecoderNet.register("rewrite_transformer_decoder")
class RewriteTransformerDecoder(DecoderNet):
    """
    用于Rewrite Transformer Decoder网络
    :param decoding_dim: `int`型，表示decoder输出的维度
    :param target_embedding_dim: `int`型，表示decoder输入的向量的维度
    :param feedforward_hidden_dim: `int`型，表示内部FFN层的维度
    :param num_layers: `int`型，表示层数
    :param num_attention_heads: `int`型，表示attention heads的数量
    :param use_positional_encoding: `bool`型，表示是否添加position embedding
    :param positional_encoding_max_steps: `int`型，表示position encoding的最大长度
    :param dropout_prob: `float`型，表示内部线性层dropout的概率
    :param residual_dropout_prob: `float`型，表示残差网络的dropout概率
    """
    def __init__(self,
                 decoding_dim: int,
                 target_embedding_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 positional_encoding_max_steps: int = 512,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2) -> None:
        super(RewriteTransformerDecoder, self).__init__(decoding_dim=decoding_dim,
                                                        target_embedding_dim=target_embedding_dim,
                                                        decodes_parallel=True)
        # 多头注意力层
        attn = MultiHeadedAttentionWithAttention(num_attention_heads,
                                                 decoding_dim)
        # FFN层
        feed_forward = PositionwiseFeedForward(decoding_dim,
                                               feedforward_hidden_dim,
                                               dropout_prob)
        # self._embed_scale = math.sqrt(decoding_dim)
        # Position Embedding层
        self._positional_embedder = PositionalEncoding(
            decoding_dim, positional_encoding_max_steps) if use_positional_encoding else None
        self._dropout = nn.Dropout(dropout_prob)
        # 定义解码器结构
        self._attention = RWDecoder(
            RWDecoderLayer(
                copy.deepcopy(attn),
                copy.deepcopy(attn),
                copy.deepcopy(attn),
                feed_forward,
                decoding_dim,
                residual_dropout_prob),
            num_layers)

    @overrides
    def init_decoder_state(self,
                           encoder_out: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        return {}

    @overrides
    def forward(self,
                previous_state: Dict[str, torch.Tensor],
                encoder_outputs: Dict[str, torch.Tensor],
                source_mask: Dict[str, torch.Tensor],
                previous_steps_predictions: torch.Tensor,
                previous_steps_mask: Optional[torch.Tensor] = None):
        context_output = encoder_outputs["context_output"]
        query_output = encoder_outputs["query_output"]
        context_mask = source_mask["context_mask"]
        query_mask = source_mask["query_mask"]
        # expand two src mask [b, 1, _len]
        context_mask = context_mask.unsqueeze(1)
        query_mask = query_mask.unsqueeze(1)

        # decoder self attention mask [1, dec_len, dec_len]
        # 上三角mask
        future_mask = subsequent_mask(
            previous_steps_predictions.size(1),
            device=previous_steps_mask.device).type_as(
            context_mask.data)

        if previous_steps_mask is None:
            dec_mask = future_mask
        else:
            # 最终的mask矩阵
            dec_mask = previous_steps_mask.unsqueeze(1) & future_mask

        if self._positional_embedder:
            previous_steps_predictions = self._positional_embedder(
                previous_steps_predictions)

        dec_embed = self._dropout(previous_steps_predictions)
        dec_output, context_attn, query_attn, x_context, x_query = self._attention(
            dec_embed, context_output, query_output, context_mask, query_mask, dec_mask)
        # dec_output: [B, dec_len, d_model]
        # context_attn: [B, dec_len, context_len]
        # query_attn: [B, dec_len, query_len]
        # x_context: [B, dec_len, d_model]
        # x_query: [B, dec_len, d_model]
        return dec_output, context_attn, query_attn, x_context, x_query


class MultiHeadedAttentionWithAttention(nn.Module):
    """多头注意力层"""
    def __init__(self, num_heads: int, input_dim: int) -> None:
        super(MultiHeadedAttentionWithAttention, self).__init__()
        assert input_dim % num_heads == 0, "input_dim must be a multiple of num_heads"
        # assume d_v equals d_k
        self.d_k = input_dim // num_heads
        self.num_heads = num_heads
        # These linear layers project from d_model to h*d_k
        self.linears = util.clone(nn.Linear(input_dim, input_dim, bias=False), 4)

    @overrides
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor,
                                                    torch.Tensor]:
        if mask is not None:
            # Usually the value mask
            # [B, num_heads, _len, _len]
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1).expand([-1, self.num_heads, -1, -1])
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1).unsqueeze(1).expand(
                    [-1, self.num_heads, query.size(1), -1])

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # [B, num_heads, _len, d_k]
        query, key, value = [
            layer(x) for layer, x in zip(self.linears, (query, key, value))]
        query = query.view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        # [B, num_heads, dec_len, d_k]    [B, num_heads, dec_len, _len]
        x, attn_prob = attention(query, key, value, mask=mask, dropout=None)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.num_heads * self.d_k)

        return self.linears[-1](x), attn_prob


class RWDecoderLayer(nn.Module):
    """
    一个Decoder层
    :param self_attn: 自注意力层
    :param context_attn: 用于focus context的注意力层
    :param query_attn: 用于focus query的注意力层
    :param feed_forward: FFN层
    :param d_model: `int`型，表示encoder和decoder输出向量的维度
    :param dropout_rate: `float`型，dropout的概率
    """
    def __init__(self,
                 self_attn: MultiHeadedAttentionWithAttention,
                 context_attn: MultiHeadedAttentionWithAttention,
                 query_attn: MultiHeadedAttentionWithAttention,
                 feed_forward: F,
                 d_model: int,
                 dropout_rate: float) -> None:
        super(RWDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.context_attn = context_attn
        self.query_attn = query_attn
        self.linear = nn.Linear(2 * d_model, d_model, bias=False)
        self.feed_forward = feed_forward
        self.norms = util.clone(nn.LayerNorm(d_model, eps=1e-6), 3)
        self.dropout = nn.Dropout(dropout_rate)

    @overrides
    def forward(self, x: torch.Tensor,
                context_output: torch.Tensor,
                query_output: torch.Tensor,
                context_mask: torch.Tensor,
                query_mask: torch.Tensor,
                tgt_mask: torch.Tensor):
        # 先进行norm
        x = self.norms[0](x)
        # dec_input self attention
        x, _ = self.self_attn(x, x, x, tgt_mask)
        # add & norm
        x = x + self.dropout(x)
        x = self.norms[1](x)
        # tgt-src cross attention
        # [B, dec_len, d_model]和[B, num_heads, dec_len, _len]
        x_context, context_attn = self.context_attn(
            x, context_output, context_output, context_mask)
        x_query, query_attn = self.query_attn(
            x, query_output, query_output, query_mask)
        # trans to the size d_model
        x_src = self.linear(torch.cat([x_context, x_query], dim=-1))
        # add & norm
        x = x + self.dropout(x_src)
        x = self.norms[2](x)
        # ffn
        x = self.feed_forward(x)
        return x, context_attn, query_attn, x_context, x_query


class RWDecoder(nn.Module):
    """
    将单个解码器层组合成多层结构
    layer: `RWDecoderLayer`类型的解码器层
    num_layers: `int`型，表示解码器层的数量
    """
    def __init__(self, layer: nn.Module, num_layers: int) -> None:
        super(RWDecoder, self).__init__()
        self.layers = util.clone(layer, num_layers)

    def forward(self, x: torch.Tensor,
                context_output: torch.Tensor,
                query_output: torch.Tensor,
                context_mask: torch.Tensor,
                query_mask: torch.Tensor,
                tgt_mask: torch.Tensor):
        # 逐层计算
        for layer in self.layers:
            x, context_attn, query_attn, x_context, x_query = layer(
                x, context_output, query_output, context_mask, query_mask, tgt_mask)
        return x, context_attn, query_attn, x_context, x_query
