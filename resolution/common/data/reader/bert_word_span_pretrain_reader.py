# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-25
import logging
import itertools
from typing import Dict, List, Optional

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, LabelField, SequenceLabelField, MetadataField

from resolution.common.data.tokenizer import TokenAdd
from resolution.common.data.reader.bert_word_span_resolution_reader import BertWordSpanResolutionReader

logger = logging.getLogger(__name__)


@DatasetReader.register("bert_word_span_pretrain")
class BertWordSpanPretrainReader(BertWordSpanResolutionReader):
    def text_to_instance(self,
                         context: List[List[TokenAdd]],
                         query: List[TokenAdd],
                         rewrite: Optional[List[str]] = None,
                         mask_label: Optional[List[int]] = None,
                         start_label: Optional[List[int]] = None,
                         end_label: Optional[List[int]] = None,
                         **kwargs) -> Instance:
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

            fields['mask_label'] = mask_label_field
            metadata['rewrite_tokens'] = rewrite
            # 判断query和rewrite是否相等
            # query要去除最后一个sep
            query_string = "".join(metadata["query_tokens"][:-1])
            rewrite_string = "".join(metadata["rewrite_tokens"])
            if query_string == rewrite_string:
                # 说明不需要改写
                cls_label = 0
            else:
                cls_label = 1
            fields['cls_label'] = LabelField(label=cls_label,
                                             skip_indexing=True)
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)
