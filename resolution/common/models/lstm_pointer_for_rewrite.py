# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-07
"""
主要用于LSTM-Encoder-Decoder的结构
"""
from typing import Dict, Optional, List, Any, Union, Tuple
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from overrides import overrides

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.metrics import Metric
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import get_mask_from_sequence_lengths

from resolution.common.models.bert_pointer_for_rewrite import BertPointerForRewrite
from resolution.common.modules.decoders import DecoderNet


TupleTensor = Tuple[torch.Tensor, torch.Tensor]


@Model.register("lstm_pointer_for_rewrite")
class LSTMPointerForRewrite(BertPointerForRewrite):
    """
    LSTM-Encoder-Decoder结构
    :param embedding_size: `int`型，表示embedder输出的维度
    :param encoder_hidden_size: `int`型，表示encoder隐层的维度
    :param encoder_num_layers: `int`型，表示encoder的层数
    :param dropout_rate: `float`型，表示内部dropout的概率
    """
    def __init__(self,
                 vocab: Vocabulary,
                 embedding_size: int,
                 encoder_hidden_size: int,
                 encoder_num_layers: int,
                 decoder: DecoderNet,
                 decoder_type: str = "lstm",
                 decoder_num_layers: int = 1,
                 share_decoder_params: bool = True,  # only valid when decoder_type == `transformer`
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
                 dropout_rate: float = 0.1,
                 seed: int = 42,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None):
        # 初始化vocab和regularizer
        Model.__init__(self, vocab, regularizer)

        # ----------- 定义embedding和编码器 ---------------
        # 获取单词序列的embedding
        self._text_field_embedder = text_field_embedder

        # 定义编码器
        self.encoder = torch.nn.LSTM(input_size=embedding_size,
                                     hidden_size=encoder_hidden_size,
                                     num_layers=encoder_num_layers,
                                     batch_first=True,
                                     dropout=dropout_rate,
                                     bidirectional=True)
        self.encoder_num_layers = encoder_num_layers
        # 由于编码器是双向的，而解码器是单向的
        # 所有将编码器的输出转换成单向的维度
        self.bi2uni_dec_init_state = torch.nn.Linear(2 * encoder_hidden_size,
                                                     encoder_hidden_size)

        self.encoder_output_dim = encoder_hidden_size

        # ------------- 通用初始化过程 ---------------------
        self.common_init(self.encoder_output_dim, decoder, decoder_type, decoder_num_layers,
                         share_decoder_params, start_token, end_token, index_name, beam_size,
                         min_dec_len, max_dec_len, coverage_factor, device, metrics,
                         valid_metric_keys, seed, initializer)

        # -------------- 不同编码器不同的初始化过程 ---------------
        # 获取embedding的维度
        embedding_size = self._text_field_embedder.get_output_dim()
        self.turn_embedding = torch.nn.Embedding(max_turn_len, embedding_size)

    @overrides
    def _get_embeddings(self,
                        ids: Union[TextFieldTensors, torch.Tensor],
                        turns: Optional[torch.Tensor] = None):
        # 如果是预测结果，则是torch.Tensor类型
        # 需要转换成TextFieldTensors
        if isinstance(ids, torch.Tensor):
            ids = {"tokens": {"tokens": ids}}
        # 得到embedding
        word_embed = self._text_field_embedder(ids)

        # add turn embeddings
        if turns is not None:
            word_embed += self.turn_embedding(turns.to(torch.long))
        return word_embed

    @overrides
    def _run_encoder(self,
                     context_embed: torch.Tensor,
                     query_embed: torch.Tensor,
                     context_len: torch.Tensor,
                     query_len: torch.Tensor):
        """
        LSTM编码器的过程
        :param context_embed: [B, context_len, embedding_size]
        :param query_embed: [B, query_len, embedding_size]
        :param context_len: [B, ]
        :param query_len: [B, ]
        :return:
        """
        b, max_context_len, _ = context_embed.size()
        max_query_len = query_embed.size(1)
        # 专门用于LSTM计算输出
        context_pack_embed = pack_padded_sequence(context_embed,
                                                  lengths=context_len,
                                                  batch_first=True,
                                                  enforce_sorted=False)
        query_pack_embed = pack_padded_sequence(query_embed,
                                                lengths=query_len,
                                                batch_first=True,
                                                enforce_sorted=False)
        # output: [B, context_len, direction*hidden_size]
        # h_n and c_n: [encoder_num_layers*num_directions, B, hidden_size]
        context_pack_output, (h_n, c_n) = self.encoder(context_pack_embed)
        query_pack_output, (h_n, c_n) = self.encoder(query_pack_embed, (h_n, c_n))

        h_n = h_n.view(self.encoder_num_layers, 2, b, self.encoder_output_dim)
        # [B, 2 * hidden_size]
        h_n = h_n[-1].transpose(0, 1).contiguous().view(b, 2 * self.encoder_output_dim).contiguous()
        c_n = c_n.view(self.encoder_num_layers, 2, b, self.encoder_output_dim)
        # [B, 2 * hidden_size]
        c_n = c_n[-1].transpose(0, 1).contiguous().view(b, 2 * self.encoder_output_dim).contiguous()

        # [B, _len, 2 * hidden_size]
        context_output, _ = pad_packed_sequence(context_pack_output,
                                                batch_first=True,
                                                total_length=max_context_len)
        query_output, _ = pad_packed_sequence(query_pack_output,
                                              batch_first=True,
                                              total_length=max_query_len)
        # 由于编码器是双向的，解码器是单向的
        # 所以需要将编码器输出的维度进行转换
        context_output = self.bi2uni_dec_init_state(context_output)
        query_output = self.bi2uni_dec_init_state(query_output)
        # 如果decoder是LSTM，则需要对初始化状态进行转化
        if self.params["decoder_type"] == "lstm":
            h_n = self.bi2uni_dec_init_state(h_n)
            c_n = self.bi2uni_dec_init_state(c_n)
            h_dec_init = h_n.unsqueeze(dim=0).expand(
                self.decoder_num_layers, -1, -1).contiguous()
            c_dec_init = c_n.unsqueeze(dim=0).expand(
                self.decoder_num_layers, -1, -1).contiguous()
            dec_init_state = (h_dec_init, c_dec_init)
        else:
            dec_init_state = None
        return context_output, query_output, dec_init_state

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
        """前向传播的过程，和基函数基本一致"""
        context_token_ids = context_ids[self._index_name]["tokens"]
        query_token_ids = query_ids[self._index_name]["tokens"]

        # get the extended context and query ids
        extend_context_ids = context_token_ids + extend_context_ids.to(dtype=torch.long)
        extend_query_ids = query_token_ids + extend_query_ids.to(dtype=torch.long)

        # ------------------ 编码器计算输出 ----------------
        # 计算context和query的embedding
        context_embed = self._get_embeddings(context_ids, context_turn)
        query_embed = self._get_embeddings(query_ids, query_turn)
        max_context_len = context_embed.size(1)
        max_query_len = query_embed.size(1)
        # 计算mask
        context_mask = get_mask_from_sequence_lengths(context_len,
                                                      max_length=max_context_len)
        query_mask = get_mask_from_sequence_lengths(query_len,
                                                    max_length=max_query_len)
        context_output, query_output, dec_init_state = self._run_encoder(context_embed,
                                                                         query_embed,
                                                                         context_len,
                                                                         query_len)
        output_dict = {"metadata": metadata}
        if self.training:
            rewrite_input_token_ids = rewrite_input_ids[self._index_name]["tokens"]
            max_rewrite_len = rewrite_input_token_ids.size(1)
            rewrite_input_mask = get_mask_from_sequence_lengths(rewrite_len,
                                                                max_length=max_rewrite_len)
            rewrite_target_ids = rewrite_target_ids[self._index_name]["tokens"]
            rewrite_target_ids = rewrite_target_ids + extend_rewrite_ids.to(dtype=torch.long)

            # 计算embedding输出，[B, rewrite_len, embedding_size]
            rewrite_embed = self._get_embeddings(rewrite_input_ids)
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