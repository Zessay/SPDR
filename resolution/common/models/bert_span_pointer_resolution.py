# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-16
import os
import logging
import copy
from overrides import overrides
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from allennlp.modules.attention import Attention, BilinearAttention

from resolution.common.modules.token_embedders import PretrainedChineseBertMismatchedEmbedder
from resolution.common.utils import get_best_span, seed_everything
from resolution.common.metrics import RewriteEM, RestorationScore, TokenBasedBLEU, TokenBasedROUGE

logger = logging.getLogger(__name__)


@Model.register("bert_span_pointer_resolution")
class BertSpanPointerResolution(Model):
    """该模型同时预测mask位置以及span的起始位置"""
    def __init__(self,
                 vocab: Vocabulary,
                 model_name: str = None,
                 start_attention: Attention = None,
                 end_attention: Attention = None,
                 text_field_embedder: TextFieldEmbedder = None,
                 task_pretrained_file: str = None,
                 neg_sample_ratio: float = 0.0,
                 max_turn_len: int = 3,
                 start_token: str = "[CLS]",
                 end_token: str = "[SEP]",
                 index_name: str = "bert",
                 eps: float = 1e-8,
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
        self._neg_sample_ratio = neg_sample_ratio
        self._start_token = start_token
        self._end_token = end_token
        self._index_name = index_name
        self._initializer = initializer

        linear_input_size = self._text_field_embedder.get_output_dim()
        # 使用attention的方法
        self.start_attention = start_attention or BilinearAttention(
            vector_dim=linear_input_size, matrix_dim=linear_input_size)
        self.end_attention = end_attention or BilinearAttention(
            vector_dim=linear_input_size, matrix_dim=linear_input_size)
        # mask的指标，主要考虑F-score，而且我们更加关注`1`的召回率
        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._rewrite_em = RewriteEM(valid_keys="semr,nr_semr,re_semr")
        self._restore_score = RestorationScore(compute_restore_tokens=True)
        self._metrics = [TokenBasedBLEU(mode="1,2"), TokenBasedROUGE(mode="1r,2r")]
        self._eps = eps

        self._initializer(self.start_attention)
        self._initializer(self.end_attention)

        # 加载其他任务预训练的模型
        if task_pretrained_file is not None and os.path.isfile(task_pretrained_file):
            logger.info("loading related task pretrained weights...")
            self.load_state_dict(torch.load(task_pretrained_file), strict=False)

    def _calc_loss(self,
                   span_start_logits: torch.Tensor,
                   span_end_logits: torch.Tensor,
                   use_mask_label: torch.Tensor,
                   start_label: torch.Tensor,
                   end_label: torch.Tensor,
                   best_spans: torch.Tensor):
        batch_size = start_label.size(0)
        # 常规loss
        loss_fct = nn.CrossEntropyLoss(reduction="none",
                                       ignore_index=-1)
        # --- 计算start和end标签对应的loss ---
        # 选择出mask_label等于1的位置对应的start和end的结果
        # [B_mask, ]
        span_start_label = start_label.masked_select(
            use_mask_label.to(dtype=torch.bool))
        span_end_label = end_label.masked_select(
            use_mask_label.to(dtype=torch.bool))
        # mask掉大部分为0的标签来计算准确率
        train_span_mask = (span_start_label != -1)

        # [B_mask, 2]
        answer_spans = torch.stack([span_start_label, span_end_label], dim=-1)
        self._span_accuracy(best_spans, answer_spans,
                            train_span_mask.unsqueeze(-1).expand_as(best_spans))

        # -- 计算start_loss --
        start_losses = loss_fct(span_start_logits, span_start_label)
        start_loss = torch.sum(start_losses) / batch_size
        # 对loss的值进行检查
        big_constant = min(torch.finfo(start_loss.dtype).max, 1e9)
        if torch.any(start_loss > big_constant):
            logger.critical("Start loss too high (%r)", start_loss)
            logger.critical("span_start_logits: %r", span_start_logits)
            logger.critical("span_start: %r", span_start_label)
            assert False

        # -- 计算end_loss --
        end_losses = loss_fct(span_end_logits, span_end_label)
        end_loss = torch.sum(end_losses) / batch_size
        if torch.any(end_loss > big_constant):
            logger.critical("End loss too high (%r)", end_loss)
            logger.critical("span_end_logits: %r", span_end_logits)
            logger.critical("span_end: %r", span_end_label)
            assert False

        span_loss = (start_loss + end_loss) / 2

        self._span_start_accuracy(span_start_logits, span_start_label, train_span_mask)
        self._span_end_accuracy(span_end_logits, span_end_label, train_span_mask)

        loss = span_loss
        return loss

    def _get_rewrite_result(self,
                            use_mask_label: torch.Tensor,
                            best_spans: torch.Tensor,
                            query_lens: torch.Tensor,
                            context_lens: torch.Tensor,
                            metadata: List[Dict[str, Any]]):
        # 将两个标签转换成numpy类型
        # [B, query_len]
        use_mask_label = use_mask_label.detach().cpu().numpy()
        # [B_mask, 2]
        best_spans = best_spans.detach().cpu().numpy().tolist()

        predict_rewrite_results = []
        for cur_query_len, cur_context_len, cur_query_mask_labels, mdata in zip(
                query_lens, context_lens, use_mask_label, metadata):
            context_tokens = mdata['context_tokens']
            query_tokens = mdata['query_tokens']
            cur_rewrite_result = copy.deepcopy(query_tokens)
            already_insert_tokens = 0  # 记录已经插入的tokens的数量
            already_insert_min_start = cur_context_len  # 表示当前已经添加过的信息的最小的start
            already_insert_max_end = 0     # 表示当前已经添加过的信息的最大的end
            # 遍历当前mask的所有标签，如果标签为1，则计算对应的span_string
            for i in range(cur_query_len):
                cur_mask_label = cur_query_mask_labels[i]
                # 只有当预测的label为1时，才进行补充
                if cur_mask_label:
                    predict_start, predict_end = best_spans.pop(0)

                    # 如果都为0则继续
                    if predict_start == 0 and predict_end == 0:
                        continue
                    # 如果start大于长度，则继续
                    if predict_start >= cur_context_len:
                        continue
                    # 如果当前想要插入的信息，在之前已经插入过信息的内部，则不再插入
                    if predict_start >= already_insert_min_start and predict_end <= already_insert_max_end:
                        continue
                    # 对位置进行矫正
                    if predict_start < 0 or context_tokens[predict_start] == self._start_token:
                        predict_start = 1

                    if predict_end >= cur_context_len:
                        predict_end = cur_context_len - 1

                    # 获取预测的span
                    predict_span_tokens = context_tokens[predict_start:predict_end + 1]
                    # 更新已经插入的最小的start和最大的end
                    if predict_start < already_insert_min_start:
                        already_insert_min_start = predict_start
                    if predict_end > already_insert_max_end:
                        already_insert_max_end = predict_end
                    # 再对预测的span按照要求进行矫正，只取end_token之前的所有tokens
                    try:
                        index = predict_span_tokens.index(self._end_token)
                        predict_span_tokens = predict_span_tokens[:index]
                    except BaseException:
                        pass

                    # 获取当前span插入的位置
                    # 如果是要插入到当前位置后面，则需要+1
                    # 如果是要插入到当前位置前面，则不需要
                    cur_insert_index = i + already_insert_tokens
                    cur_rewrite_result = cur_rewrite_result[:cur_insert_index] + \
                        predict_span_tokens + cur_rewrite_result[cur_insert_index:]
                    # 记录插入的tokens的数量
                    already_insert_tokens += len(predict_span_tokens)

            cur_rewrite_result = cur_rewrite_result[:-1]
            # 不再以list of tokens的形式
            # 而是以string的形式去计算
            cur_rewrite_string = "".join(cur_rewrite_result)
            rewrite_tokens = mdata.get("rewrite_tokens", None)
            if rewrite_tokens is not None:
                rewrite_string = "".join(rewrite_tokens)
                # 去除[SEP]这个token
                query_string = "".join(query_tokens[:-1])
                self._rewrite_em(cur_rewrite_string,
                                 rewrite_string,
                                 query_string)
                # 额外增加的指标
                for metric in self._metrics:
                    metric(cur_rewrite_result, rewrite_tokens)
                # 获取restore_tokens并计算对应的指标
                restore_tokens = mdata.get("restore_tokens", None)
                self._restore_score(cur_rewrite_result, rewrite_tokens,
                                    queries=query_tokens[:-1],
                                    restore_tokens=restore_tokens)


            predict_rewrite_results.append("".join(cur_rewrite_result))
        return predict_rewrite_results

    @overrides
    def forward(self,
                context_ids: TextFieldTensors,
                query_ids: TextFieldTensors,
                context_lens: torch.Tensor,
                query_lens: torch.Tensor,
                mask_label: Optional[torch.Tensor] = None,
                start_label: Optional[torch.Tensor] = None,
                end_label: Optional[torch.Tensor] = None,
                metadata: List[Dict[str, Any]] = None):
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
            embedder_outputs = self._text_field_embedder(
                **dialogue_ids[self._index_name])

        # get the outputs of the query and context
        # [B, _len, embed_size]
        context_last_layer = embedder_outputs[:, :context_len].contiguous()
        query_last_layer = embedder_outputs[:, context_len:].contiguous()

        # ------- 计算span预测的结果 -------
        # 我们想要知道query中的每一个mask位置的token后面需要补充的内容
        # 也就是其对应的context中span的start和end的位置
        # 同理，将context扩展成 [b, query_len, context_len, embed_size]
        context_last_layer = context_last_layer.unsqueeze(
            dim=1).expand(-1, query_len, -1, -1).contiguous()
        # [b, query_len, context_len]
        context_expand_mask = context_mask.unsqueeze(
            dim=1).expand(-1, query_len, -1).contiguous()

        # 将上面3个部分拼接在一起
        # 这里表示query中所有的position
        span_embed_size = context_last_layer.size(-1)

        if self.training and self._neg_sample_ratio > 0.0:
            # 对mask中0的位置进行采样
            # [B*query_len, ]
            sample_mask_label = mask_label.view(-1)
            # 获取展开之后的长度以及需要采样的负样本的数量
            mask_length = sample_mask_label.size(0)
            mask_sum = int(torch.sum(sample_mask_label).item() * self._neg_sample_ratio)
            mask_sum = max(10, mask_sum)
            # 获取需要采样的负样本的索引
            neg_indexes = torch.randint(low=0, high=mask_length, size=(mask_sum, ))
            # 限制在长度范围内
            neg_indexes = neg_indexes[:mask_length]
            # 将负样本对应的位置mask置为1
            sample_mask_label[neg_indexes] = 1
            # [B, query_len]
            use_mask_label = sample_mask_label.view(-1, query_len).to(dtype=torch.bool)
            # 过滤掉query中pad的部分, [B, query_len]
            use_mask_label = use_mask_label & query_mask
            span_mask = use_mask_label.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # 选择context部分可以使用的内容
            # [B_mask, context_len, span_embed_size]
            span_context_matrix = context_last_layer.masked_select(
                span_mask).view(-1, context_len, span_embed_size).contiguous()
            # 选择query部分可以使用的向量
            span_query_vector = query_last_layer.masked_select(
                span_mask.squeeze(dim=-1)).view(-1, span_embed_size).contiguous()
            span_context_mask = context_expand_mask.masked_select(
                span_mask.squeeze(dim=-1)).view(-1, context_len).contiguous()
        else:
            use_mask_label = query_mask
            span_mask = use_mask_label.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # 选择context部分可以使用的内容
            # [B_mask, context_len, span_embed_size]
            span_context_matrix = context_last_layer.masked_select(
                span_mask).view(-1, context_len, span_embed_size).contiguous()
            # 选择query部分可以使用的向量
            span_query_vector = query_last_layer.masked_select(
                span_mask.squeeze(dim=-1)).view(-1, span_embed_size).contiguous()
            span_context_mask = context_expand_mask.masked_select(
                span_mask.squeeze(dim=-1)).view(-1, context_len).contiguous()

        # 得到span属于每个位置的logits
        # [B_mask, context_len]
        span_start_probs = self.start_attention(span_query_vector,
                                                span_context_matrix,
                                                span_context_mask)
        span_end_probs = self.end_attention(span_query_vector,
                                            span_context_matrix,
                                            span_context_mask)

        span_start_logits = torch.log(span_start_probs + self._eps)
        span_end_logits = torch.log(span_end_probs + self._eps)

        # [B_mask, 2]，最后一个维度第一个表示start的位置，第二个表示end的位置
        best_spans = get_best_span(span_start_logits, span_end_logits)
        # 计算得到每个best_span的分数
        best_span_scores = (torch.gather(span_start_logits, 1, best_spans[:, 0].unsqueeze(1))
                            + torch.gather(span_end_logits, 1, best_spans[:, 1].unsqueeze(1)))
        # [B_mask, ]
        best_span_scores = best_span_scores.squeeze(1)

        # 将重要的信息写入到输出中
        output_dict = {
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_spans": best_spans,
            "best_span_scores": best_span_scores
        }

        # 如果存在标签，则使用标签计算loss
        if start_label is not None:
            loss = self._calc_loss(span_start_logits,
                                   span_end_logits,
                                   use_mask_label,
                                   start_label,
                                   end_label,
                                   best_spans)
            output_dict["loss"] = loss
        if metadata is not None:
            predict_rewrite_results = self._get_rewrite_result(use_mask_label,
                                                               best_spans,
                                                               query_lens,
                                                               context_lens,
                                                               metadata)
            output_dict['rewrite_results'] = predict_rewrite_results
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        metrics["span_acc"] = self._span_accuracy.get_metric(reset)
        for metric in self._metrics:
            metrics.update(metric.get_metric(reset))
        metrics.update(self._rewrite_em.get_metric(reset))
        metrics.update(self._restore_score.get_metric(reset))
        return metrics

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_output_dict = {}
        new_output_dict["rewrite_results"] = output_dict["rewrite_results"]
        return new_output_dict
