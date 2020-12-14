# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-18
import logging
from overrides import overrides
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from resolution.common.modules.token_embedders import PretrainedChineseBertMismatchedEmbedder
from resolution.common.utils import seed_everything

logger = logging.getLogger(__name__)


@Model.register("bert_span_pretrain")
class BertSpanPretrain(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 model_name: str = None,
                 text_field_embedder: TextFieldEmbedder = None,
                 max_turn_len: int = 3,
                 start_token: str = "[CLS]",
                 end_token: str = "[SEP]",
                 index_name: str = "bert",
                 mask_task: bool = True,
                 cls_task: bool = True,
                 seed: int = 42,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None):
        super().__init__(vocab, regularizer)
        if model_name is None and text_field_embedder is None:
            raise ValueError(
                f"`model_name` and `text_field_embedder` can't both equal to None.")
        # 单纯的resolution任务，只需要返回最后一层的embedding表征即可
        self._text_field_embedder = text_field_embedder or PretrainedChineseBertMismatchedEmbedder(
            model_name, return_all=False, output_hidden_states=False, max_turn_length=max_turn_len)

        seed_everything(seed)
        self._start_token = start_token
        self._end_token = end_token
        self._cls_task = cls_task
        self._mask_task = mask_task

        self._index_name = index_name
        self._initializer = initializer

        linear_input_size = self._text_field_embedder.get_output_dim()
        # 线性层
        # 判断是否需要填充
        self._cls_linear = nn.Linear(linear_input_size, 2)
        self.cls_acc = CategoricalAccuracy()
        # 判断要填充的位置
        self._mask_linear = nn.Linear(linear_input_size, 2)
        self.mask_acc = CategoricalAccuracy()

    @overrides
    def forward(self,
                context_ids: TextFieldTensors,
                query_ids: TextFieldTensors,
                context_lens: torch.Tensor,
                query_lens: torch.Tensor,
                mask_label: Optional[torch.Tensor] = None,
                cls_label: Optional[torch.Tensor] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # concat the context and query to the encoder
        # get the indexers first
        indexers = context_ids.keys()
        dialogue_ids = {}

        # 获取context和query的长度
        context_len = torch.max(context_lens).item()
        query_len = torch.max(query_lens).item()

        # [B, _len]
        context_mask = get_mask_from_sequence_lengths(
            context_lens, context_len)
        query_mask = get_mask_from_sequence_lengths(query_lens, query_len)
        for indexer in indexers:
            # get the various variables of context and query
            dialogue_ids[indexer] = {}
            for key in context_ids[indexer].keys():
                context = context_ids[indexer][key]
                query = query_ids[indexer][key]
                # concat the context and query in the length dim
                dialogue = torch.cat([context, query], dim=1)
                dialogue_ids[indexer][key] = dialogue

        # get the outputs of the dialogue
        if isinstance(self._text_field_embedder, TextFieldEmbedder):
            embedder_outputs = self._text_field_embedder(dialogue_ids)
        else:
            embedder_outputs = self._text_field_embedder(**dialogue_ids[self._index_name])

        # get the outputs of the query and context
        # [B, _len, embed_size]
        context_last_layer = embedder_outputs[:, :context_len].contiguous()
        query_last_layer = embedder_outputs[:, context_len:].contiguous()

        output_dict = {}
        # --------- cls任务：判断是否需要改写 ------------------
        if self._cls_task:
            # 获取cls表征, [B, embed_size]
            cls_embed = context_last_layer[:, 0]
            # 经过线性层分类, [B, 2]
            cls_logits = self._cls_linear(cls_embed)
            output_dict["cls_logits"] = cls_logits
        else:
            cls_logits = None

        # --------- mask任务：判断query中需要填充的位置 -----------
        if self._mask_task:
            # 经过线性层，[B, _len, 2]
            mask_logits = self._mask_linear(query_last_layer)
            output_dict["mask_logits"] = mask_logits
        else:
            mask_logits = None

        if cls_label is not None:
            output_dict["loss"] = self._calc_loss(cls_label,
                                                  mask_label,
                                                  cls_logits,
                                                  mask_logits,
                                                  query_mask)

        return output_dict

    def _calc_loss(self,
                   cls_label: torch.Tensor,
                   mask_label: torch.Tensor,
                   cls_logits: Optional[torch.Tensor] = None,
                   mask_logits: Optional[torch.Tensor] = None,
                   query_mask: Optional[torch.Tensor] = None):
        batch_size, query_len = query_mask.size()
        # 定义loss
        loss_fct = nn.CrossEntropyLoss(reduction="none",
                                       ignore_index=-1)
        # --------- 计算cls任务的loss -------------
        if cls_logits is not None:
            cls_losses = loss_fct(cls_logits, cls_label)
            cls_loss = torch.sum(cls_losses) / batch_size
            self.cls_acc(cls_logits, cls_label)
            loss = cls_loss

        # --------- 计算mask任务的loss -------------
        if mask_logits is not None:
            mask_label = mask_label.view(-1)
            mask_logits = mask_logits.view(batch_size*query_len, 2)
            # [B*query_len, ]
            mask_losses = loss_fct(mask_logits, mask_label)
            query_mask = query_mask.view(-1)
            mask_losses = mask_losses * query_mask
            mask_loss = torch.sum(mask_losses) / batch_size
            self.mask_acc(mask_logits, mask_label, query_mask.to(torch.bool))
            loss = mask_loss

        # 得到最终的loss
        if cls_logits is not None and mask_logits is not None:
            loss = cls_loss + mask_loss

        return loss

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        metrics["cls_acc"] = self.cls_acc.get_metric(reset)
        metrics["mask_acc"] = self.mask_acc.get_metric(reset)
        return metrics