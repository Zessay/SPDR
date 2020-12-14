# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-07
"""
主要用于Transformer Encoder-Decoder的结构
"""
from typing import Dict, Optional, List, Any, Union, Tuple
import torch
from overrides import overrides

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.metrics import Metric
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import get_mask_from_sequence_lengths, add_positional_features
from allennlp.modules.attention import DotProductAttention

from resolution.common.models.bert_pointer_for_rewrite import BertPointerForRewrite
from resolution.common.modules.decoders import DecoderNet


TupleTensor = Tuple[torch.Tensor, torch.Tensor]


@Model.register("transformer_pointer_for_rewrite")
class TransformerPointerForRewrite(BertPointerForRewrite):
    """
    Transformer Encoder-Decoder的结构
    理论上解码器也可以是LSTM
    :param encoder: `Seq2SeqEncoder`型，表示解码器
    :param encoder_num_layers: `int`型，表示编码器的层数
    :param share_encoder_params: `bool`型，表示是否共享encoder参数
    """
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 decoder: DecoderNet,
                 decoder_type: str = "lstm",
                 encoder_num_layers: int = 1,
                 decoder_num_layers: int = 1,
                 share_encoder_params: bool = True,
                 share_decoder_params: bool = True,
                 text_field_embedder: TextFieldEmbedder = None,
                 start_token: str = "[CLS]",
                 end_token: str = "[SEP]",
                 index_name: str = "tokens",
                 beam_size: int = 4,
                 max_turn_len: int = 3,
                 min_dec_len: int = 4,
                 max_dec_len: int = 30,
                 coverage_factor: float = 0.0,
                 device: Union[int, str, List[int]] = -1,
                 metrics: Optional[List[Metric]] = None,
                 valid_metric_keys: List[str] = None,
                 seed: int = 42,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None):
        # 初始化vocab和regularizer
        Model.__init__(self, vocab, regularizer)

        # ---------- 定义embedding和编码器 -----------------
        # 获取单词序列的embedding
        # 通常是Embedding这个类
        self._text_field_embedder = text_field_embedder

        # 定义编码器
        self.encoder = encoder
        # 获取编码器的输出维度
        self.encoder_output_dim = self.encoder.get_output_dim()

        # ---------- 通用初始化过程 -------------
        self.common_init(self.encoder_output_dim, decoder, decoder_type, decoder_num_layers,
                         share_decoder_params, start_token, end_token, index_name, beam_size,
                         min_dec_len, max_dec_len, coverage_factor, device, metrics,
                         valid_metric_keys, seed, initializer)

        # --------- 不同编码器不同的初始化过程 -------
        # 获取embedding的维度
        embedding_size = self._text_field_embedder.get_output_dim()
        self.turn_embedding = torch.nn.Embedding(max_turn_len, embedding_size)

        self.encoder_num_layers = encoder_num_layers
        self._share_encoder_params = share_encoder_params
        # 如果解码器是LSTM，则需要使用attention初始化LSTM的初始状态
        # 如果编码器也是LSTM，则不需要
        if self.params["decoder_type"] == "lstm":
            self.h_query = torch.nn.Parameter(torch.randn([self.encoder_output_dim]),
                                              requires_grad=True)
            self.c_query = torch.nn.Parameter(torch.randn([self.encoder_output_dim]),
                                              requires_grad=True)
            self.init_attention = DotProductAttention()

    @overrides
    def _get_embeddings(self,
                        ids: Union[TextFieldTensors, torch.Tensor],
                        turns: Optional[torch.Tensor] = None):
        # 解码阶段，对于的输入是torch.Tensor
        # 需要转化为torch.Tensor类型
        if isinstance(ids, torch.Tensor):
            ids = {"tokens": {"tokens": ids}}
        # 得到embedding
        word_embed = self._text_field_embedder(ids)

        # add turn embeddings
        if turns is not None:
            word_embed += self.turn_embedding(turns.to(torch.long))
        return word_embed

    @overrides
    def forward(self,
                context_ids: TextFieldTensors,
                query_ids: TextFieldTensors,
                extend_context_ids: torch.Tensor,
                extend_query_ids: torch.Tensor,
                context_turn: torch.Tensor,
                query_turn: torch.Tensor,
                context_len: torch.Tensor,
                query_len: torch.Tensor,
                oovs_len: torch.Tensor,
                rewrite_input_ids: Optional[TextFieldTensors] = None,
                rewrite_target_ids: Optional[TextFieldTensors] = None,
                extend_rewrite_ids: Optional[torch.Tensor] = None,
                rewrite_len: Optional[torch.Tensor] = None,
                metadata: Optional[List[Dict[str, Any]]] = None):
        """前向传播的过程"""
        context_token_ids = context_ids[self._index_name]["tokens"]
        query_token_ids = query_ids[self._index_name]["tokens"]

        # get the extended context and query ids
        extend_context_ids = context_token_ids + extend_context_ids.to(dtype=torch.long)
        extend_query_ids = query_token_ids + extend_query_ids.to(dtype=torch.long)

        # ---------------- 编码器计算输出 ------------
        # 计算context和query的embedding
        context_embed = self._get_embeddings(context_ids, context_turn)
        query_embed = self._get_embeddings(query_ids, query_turn)
        # 计算context和query的长度
        max_context_len = context_embed.size(1)
        max_query_len = query_embed.size(1)
        # 计算mask
        context_mask = get_mask_from_sequence_lengths(context_len,
                                                      max_length=max_context_len)
        query_mask = get_mask_from_sequence_lengths(query_len,
                                                    max_length=max_query_len)
        # 计算编码器输出
        dialogue_embed = torch.cat([context_embed, query_embed], dim=1)
        dialogue_mask = torch.cat([context_mask, query_mask], dim=1)
        # 如果共享编码器参数，需要提交添加位置编码
        if self._share_encoder_params:
            dialogue_embed = add_positional_features(dialogue_embed)
        # 编码器输出 [B, dialogue_len, encoder_output_dim]
        dialogue_output = self.encoder(dialogue_embed, dialogue_mask)
        if self._share_encoder_params:
            for _ in range(self.encoder_num_layers - 1):
                dialogue_output = self.encoder(dialogue_output,
                                               dialogue_mask)
        # 计算编码结果
        # [B, context_len, *]和[B, query_len, *]
        context_output, query_output, dec_init_state = self._run_encoder(dialogue_output,
                                                                         context_mask,
                                                                         query_mask)
        output_dict = {"metadata": metadata}
        if self.training:
            rewrite_input_token_ids = rewrite_input_ids[self._index_name]["tokens"]
            # 计算rewrite的长度
            max_rewrite_len = rewrite_input_token_ids.size(1)
            rewrite_input_mask = get_mask_from_sequence_lengths(rewrite_len,
                                                                max_length=max_rewrite_len)
            rewrite_target_ids = rewrite_target_ids[self._index_name]["tokens"]
            # 计算rewrite的目标序列索引
            rewrite_target_ids = rewrite_target_ids + extend_rewrite_ids.to(dtype=torch.long)

            # 计算embedding输出，[B, rewrite_len, embedding_size]
            rewrite_embed = self._get_embeddings(rewrite_input_ids)
            # 前向传播计算loss
            new_output_dict = self._forward_step(context_output, query_output, context_mask, query_mask,
                                                 rewrite_embed, rewrite_target_ids, rewrite_len,
                                                 rewrite_input_mask, extend_context_ids,
                                                 extend_query_ids, oovs_len, dec_init_state)
            output_dict.update(new_output_dict)
        else:
            batch_hyps = self._run_inference(context_output,
                                             query_output,
                                             context_mask,
                                             query_mask,
                                             extend_context_ids,
                                             extend_query_ids,
                                             oovs_len,
                                             dec_init_state=dec_init_state)
            # get the result of each instance
            output_dict['hypothesis'] = batch_hyps
            output_dict = self.get_rewrite_string(output_dict)
            output_dict["loss"] = torch.tensor(0)

        return output_dict
