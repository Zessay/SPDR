# coding=utf-8
# @Author: 莫冉
# @Date: 2020-06-22
"""
多层LSTM解码器
"""
from typing import Tuple, Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn
from allennlp.nn import util

from resolution.common.modules.decoders import DecoderNet


@DecoderNet.register("stacked_lstm_decoder")
class StackedLstmDecoder(DecoderNet):
    """"""
    def __init__(self,
                 decoding_dim: int,
                 target_embedding_dim: int,
                 num_layers: int,
                 batch_first: bool = True,
                 dropout: float = 0.1,
                 bidirectional_input: bool = False):
        """
        多层LSTM解码器
        :param decoding_dim: `int`型，表示解码器的输出，也就是隐层的维度
        :param target_embedding_dim: `int`型，表示解码器输入embedding的维度
        :param num_layers: `int`型，表示LSTM的层数
        :param batch_first: `bool`型
        :param dropout: `float`型，表示dropout的概率
        :param bidirectional_input: `bool`型，表示编码器输出是否是双向的，只有在计算初始状态时使用
        """
        super(StackedLstmDecoder, self).__init__(decoding_dim=decoding_dim,
                                                 target_embedding_dim=target_embedding_dim,
                                                 decodes_parallel=False)
        self._decoder = nn.LSTM(input_size=target_embedding_dim,
                                hidden_size=decoding_dim,
                                num_layers=num_layers,
                                batch_first=batch_first,
                                dropout=dropout,
                                bidirectional=False)
        self._bidirectional_input = bidirectional_input
        self.decoding_dim = decoding_dim

    def init_decoder_state(self,
                           encoder_out: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:

        batch_size, _ = encoder_out['source_mask'].size()

        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(encoder_out["encoder_outputs"],
                                                             encoder_out["source_mask"],
                                                             bidirectional=self._bidirectional_input)

        return {
            # that is to say, the decoder_output_dim = encoder_output_Dim
            "decoder_hidden": final_encoder_output,
            "decoder_context": final_encoder_output.new_zeros(batch_size, self.decoding_dim)
        }

    @overrides
    def forward(self,
                previous_state: Dict[str, torch.Tensor],
                encoder_outputs: Dict[str, torch.Tensor],
                source_mask: Dict[str, torch.Tensor],
                previous_steps_predictions: torch.Tensor,
                previous_steps_mask: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor],
                                                                             torch.Tensor]:
        # 分别得到decoder初始化状态的context和hidden
        # 维度都是 [num_layers, B, decoding_dim]
        decoder_hidden = previous_state['decoder_hidden']
        decoder_context = previous_state['decoder_context']

        # shape: (batch, seq_len, target_embedding_dim)
        decoder_input = previous_steps_predictions

        decoder_output, (decoder_hidden, decoder_context) = self._decoder(
            decoder_input, (decoder_hidden, decoder_context))

        return {"decoder_hidden": decoder_hidden,
                "decoder_context": decoder_context}, decoder_output
