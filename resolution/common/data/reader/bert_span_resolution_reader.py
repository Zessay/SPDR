# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-10
import logging
import itertools
from typing import Dict, List, Optional
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, LabelField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

from resolution.common.data.tokenizer import TokenAdd, ChineseCharacterTokenizer
from resolution.common.data.token_indexer import PretrainedChineseBertMismatchedIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("bert_span_resolution")
class BertSpanResolutionReader(DatasetReader):
    """通过span预测的方式完成消解任务的DataReader"""
    def __init__(self,
                 model_name: str,
                 namespace: str = "bert_tags",
                 max_turn_len: int = 3,
                 max_length: int = 512,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 start_token: str = "[CLS]",
                 end_token: str = "[SEP]",
                 do_lowercase: bool = True,
                 never_split: List[str] = None,
                 index_name: str = "bert",
                 lazy: bool = False,
                 **kwargs):
        super().__init__(lazy, **kwargs)
        if never_split is not None:
            never_split = [start_token, end_token] + never_split
        else:
            never_split = [start_token, end_token]
        self._tokenizer = tokenizer or ChineseCharacterTokenizer(do_lowercase,
                                                                 never_split)

        self._token_indexers = token_indexers or {
            index_name: PretrainedChineseBertMismatchedIndexer(model_name, namespace)}
        self._max_turn_len = max_turn_len - 1
        self._start_token = start_token
        self._end_token = end_token
        self.max_length = max_length

    def _get_turn_ids(self,
                      context: List[List[Token]],
                      query: List[Token]):
        """Get the turn index of each token in context and query"""
        origin_turns = len(context)
        # total length of context
        total_len = sum([len(sent) for sent in context])
        context_turn = []

        if origin_turns > self._max_turn_len:
            cur_turn = self._max_turn_len - 1
            for sent in reversed(context):
                context_turn = [cur_turn] * len(sent) + context_turn
                cur_turn -= 1
                if cur_turn < 0:
                    break
            context_turn = [0] * (total_len - len(context_turn)) + context_turn
            query_turn = [self._max_turn_len] * len(query)

        else:
            # traverse from left to right
            cur_turn = 0
            for sent in context:
                context_turn += [cur_turn] * len(sent)
                cur_turn += 1
            query_turn = [cur_turn] * len(query)
        assert len(context_turn) == total_len
        return context_turn, query_turn

    def _limit_max_length(self, context: List[List[Token]], query: List[Token]):
        context_len = sum([len(turn) for turn in context])
        query_len = len(query)
        total_len = context_len + query_len
        sub_len = 0

        # 如果大于最大长度限制
        if total_len > self.max_length:
            sub_len = total_len - self.max_length
            remove_len = sub_len
            while remove_len > 0:
                first_turn = context.pop(0)
                first_turn_len = len(first_turn)
                if first_turn_len > sub_len:
                    remained_turn = [TokenAdd(self._start_token)] + first_turn[-sub_len + 1:]
                    context = [remained_turn] + context
                    remove_len = 0
                else:
                    remove_len = remove_len - first_turn_len + 1
                    context[0] = [TokenAdd(self._start_token)] + context[0]
        return context, sub_len

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # 按顺序分别是context, query和rewrite，以及mask, start和end
                    line_list = line.strip().split("\t\t")
                    context, query = line_list[0], line_list[1]
                    context, query, sub_len = self.make_input_to_instance(context, query)

                    rewrite, mask_label, start_label, end_label = None, None, None, None
                    restore_string = ""
                    # 封装rewrite，mask、start以及end部分
                    if len(line_list) > 2:
                        rewrite = line_list[2]
                        rewrite = [token.text for token in self._tokenizer.tokenize(rewrite)]
                        mask_string, start_string, end_string, *restore_strings = line_list[3:]
                        # 这里由于去除了query前面的[CLS]标志，所以自动向后顺延一位
                        # 即预测对应位置前面的span情况
                        mask_label = [int(m) for m in mask_string.split(',')][:-1]
                        start_label = [int(s) - sub_len for s in start_string.split(',')][:-1]
                        end_label = [int(e) - sub_len for e in end_string.split(',')][:-1]

                        # 将-1转化为0
                        start_label = [max(0, s) for s in start_label]
                        end_label = [max(0, e) for e in end_label]
                        # 判断是否存在restore_strings
                        if restore_strings:
                            restore_string = restore_strings[0]

                    yield self.text_to_instance(context, query, rewrite, mask_label, start_label, end_label,
                                                restore_string=restore_string)
                except Exception as e:
                    logger.info(f"read file exception: {line} | {e}")

    def make_input_to_instance(self, context: str, query: str):
        contexts = context.split("<EOS>")
        # 使用chinese-character-tokenizer并没有添加起始的special tokens
        # 而使用pretrained-tokenizer会添加起始的special tokens，如果没有修改参数
        context = [self._tokenizer.tokenize(sent) for sent in contexts]
        query = self._tokenizer.tokenize(query)
        # 在context的第一个turn前后加上special token，其他turn后面加上special token
        for i in range(len(context)):
            if i == 0:
                context[i] = [TokenAdd(text=self._start_token)] + context[i] + [TokenAdd(text=self._end_token)]
            else:
                context[i] = context[i] + [TokenAdd(text=self._end_token)]
        # 在query前后加上special token
        query = query + [TokenAdd(text=self._end_token)]

        # limit the max length of context
        context, sub_len = self._limit_max_length(context, query)

        return context, query, sub_len

    @overrides
    def text_to_instance(self,
                         context: List[List[TokenAdd]],
                         query: List[TokenAdd],
                         rewrite: Optional[List[str]] = None,
                         mask_label: Optional[List[int]] = None,
                         start_label: Optional[List[int]] = None,
                         end_label: Optional[List[int]] = None,
                         restore_string: Optional[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        # flatten context
        context_flat = list(itertools.chain(*context))
        context_turn, query_turn = self._get_turn_ids(context, query)
        # add the turn id to the context and query
        for token, turn_id in zip(context_flat, context_turn):
            token.turn_id = turn_id
            token.type_id = 0
        for token, turn_id in zip(query, query_turn):
            token.turn_id = turn_id
            token.type_id = 1
        # get the context and query field and length field
        context_tokens_field = TextField(context_flat, self._token_indexers)
        query_tokens_field = TextField(query, self._token_indexers)

        context_len_field = LabelField(label=len(context_flat),
                                       label_namespace="len_tags",
                                       skip_indexing=True)
        query_len_field = LabelField(label=len(query),
                                     label_namespace="len_tags",
                                     skip_indexing=True)

        fields['context_ids'] = context_tokens_field
        fields['query_ids'] = query_tokens_field
        fields['context_lens'] = context_len_field
        fields['query_lens'] = query_len_field

        # preserve the original text
        metadata = {
            "context_tokens": [token.text for sent in context for token in sent],
            "query_tokens": [token.text for token in query]
        }

        # get the label for train
        if rewrite:
            mask_label_field = SequenceLabelField(mask_label,
                                                  sequence_field=query_tokens_field)
            start_label_field = SequenceLabelField(start_label,
                                                   sequence_field=query_tokens_field)
            end_label_field = SequenceLabelField(end_label,
                                                 sequence_field=query_tokens_field)

            fields['mask_label'] = mask_label_field
            fields['start_label'] = start_label_field
            fields['end_label'] = end_label_field
            metadata['rewrite_tokens'] = rewrite
            restore_string = restore_string or ""
            metadata['restore_tokens'] = restore_string.split()   # 得到还原之后的tokens

        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)
