# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-12
import logging
from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer

from resolution.common.data.reader import BertSpanResolutionReader
from resolution.common.data.tokenizer import TokenAdd, JiebaTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("bert_word_span_resolution")
class BertWordSpanResolutionReader(BertSpanResolutionReader):
    def __init__(self,
                 model_name: str,
                 namespace: str = "bert_tags",
                 max_turn_len: int = 3,
                 max_length: int = 512,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 context_tokenizer: Tokenizer = None,
                 query_tokenizer: Tokenizer = None,
                 start_token: str = "[CLS]",
                 end_token: str = "[SEP]",
                 do_lowercase: bool = True,
                 never_split: List[str] = None,
                 index_name: str = "bert",
                 lazy: bool = False,
                 **kwargs):
        super().__init__(model_name, namespace, max_turn_len, max_length,
                         token_indexers, context_tokenizer, start_token, end_token,
                         do_lowercase, never_split, index_name, lazy, **kwargs)

        self._query_tokenizer= query_tokenizer or WhitespaceTokenizer()
        never_split = (never_split or []) + [start_token, end_token]
        self._jieba_tokenizer = JiebaTokenizer(do_lowercase=do_lowercase,
                                               never_split=never_split)

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r", encoding="utf-8") as f:
            for line in f:
                # 按顺序依次是context（<EOS>分隔）
                # query（<\s>分隔）
                # rewrite
                # mask, start和end（用,分隔）
                try:
                    line_list = line.strip().split("\t\t")
                    context, query = line_list[0], line_list[1]
                    # 替换为空格
                    query = query.replace("<\s>", " ")
                    context, query, sub_len = self.make_input_to_instance(context, query, is_training=True)

                    rewrite, start_label, end_label = None, None, None
                    # 封装rewrite，mask/start/end
                    if len(line_list) > 2:
                        rewrite = line_list[2]
                        rewrite = [token.text for token in self._tokenizer.tokenize(rewrite)]
                        mask_string, start_string, end_string = line_list[3:]

                        # word形式的标签，我们是按照顺序预处理好的，没有包含开头的[CLS]
                        mask_label = [int(m) for m in mask_string.split(',')]
                        start_label = [int(s) - sub_len for s in start_string.split(',')]
                        end_label = [int(e) - sub_len for e in end_string.split(',')]

                        # 将-1转化为0
                        start_label = [max(0, s) for s in start_label]
                        end_label = [max(0, e) for e in end_label]

                    yield self.text_to_instance(context, query, rewrite, mask_label, start_label, end_label)
                except Exception as e:
                    logger.info(f"read file exception: {line} | {e}")

    def make_input_to_instance(self, context: str, query: str, is_training: bool = False):
        contexts = context.split("<EOS>")
        # context仍然使用标准的按字分割
        context = [self._tokenizer.tokenize(sent) for sent in contexts]
        # 如果是训练阶段，则需要使用white-space-tokenizer
        # 如果是预测阶段，则使用jieba-tokenizer
        if is_training:
            # 训练阶段
            query = self._query_tokenizer.tokenize(query)
            query = [TokenAdd(text=token.text) for token in query]
        else:
            query = self._jieba_tokenizer.tokenize(query)

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
