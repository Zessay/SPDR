# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-06
import itertools
from typing import Dict, List, Optional
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data import Vocabulary

from resolution.common.data.reader import PointerRewriteReader
from resolution.common.data.tokenizer import TokenAdd, PretrainedChineseBertTokenizer
from resolution.common.data.token_indexer import PretrainedChineseBertIndexer


@DatasetReader.register('bert_pointer_rewrite')
class BertPointerRewriteReader(PointerRewriteReader):
    """
    The dataset reader which is used for pointer rewrite network.

    # Parameters
    model_name: `str`, required
        The model name or model path of the pretrained bert model.
    vocab_path: `str`, required
        The vocab path saved by `Vocabulary` class.
    """
    def __init__(self,
                 model_name: str,
                 vocab_path: str,
                 max_enc_len: int = 512,
                 max_dec_len: int = 30,
                 max_turn_len: int = 3,
                 index_name: str = "bert",
                 namespace: str = "bert_tags",
                 start_token: str = "[CLS]",
                 end_token: str = "[SEP]",
                 pad_token: str = "[PAD]",
                 oov_token: str = "[UNK]",
                 do_lower_case: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 never_split: List[str] = None,
                 lazy: bool = False,
                 **kwargs):
        token_indexers = token_indexers or {
            index_name: PretrainedChineseBertIndexer(model_name, namespace)}
        self._pretrained_tokenizer = PretrainedChineseBertTokenizer(model_name).tokenizer
        super().__init__(vocab_path=vocab_path,
                         max_enc_len=max_enc_len,
                         max_dec_len=max_dec_len,
                         max_turn_len=max_turn_len,
                         index_name="tokens",
                         start_token=start_token,
                         end_token=end_token,
                         pad_token=pad_token,
                         oov_token=oov_token,
                         do_lower_case=do_lower_case,
                         token_indexers=token_indexers,
                         tokenizer=tokenizer,
                         never_split=never_split,
                         lazy=lazy, **kwargs)

    def _get_pretrained_token_id(self, token: TokenAdd):
        if token.text_id is None:
            token.text_id = self._pretrained_tokenizer.convert_tokens_to_ids(token.text)

        # 如果转化之后仍然为None，则转化为unk
        if token.text_id is None:
            token.text_id = self._unk_id

    @overrides
    def text_to_instance(self,
                         context: List[List[TokenAdd]],
                         query: List[TokenAdd],
                         rewrite: Optional[List[TokenAdd]] = None) -> Instance:

        context_flat = list(itertools.chain(*context))

        fields = self.get_common_field(context_flat, query, rewrite)
        # get the turn field of context and query
        context_turn_field, query_turn_field = self._get_turn_field_value(context, query)
        context_turn = context_turn_field.array
        query_turn = query_turn_field.array

        # add the turn id to context and query
        # and get the token id
        for token, turn_id in zip(context_flat, context_turn):
            # 获取token id
            self._get_pretrained_token_id(token)
            token.turn_id = turn_id
            token.type_id = 0
        for token, turn_id in zip(query, query_turn):
            self._get_pretrained_token_id(token)
            token.turn_id = turn_id
            token.type_id = 1
        # get the context and query field and length field
        context_tokens_field = TextField(context_flat, self._token_indexers)
        query_tokens_field = TextField(query, self._token_indexers)

        fields['context_ids'] = context_tokens_field
        fields['query_ids'] = query_tokens_field

        # 如果rewrite不为空，则需要对rewrite_input和rewrite_target进行转换
        if rewrite is not None:
            rewrite_input_ids_field = fields['rewrite_input_ids']
            rewrite_target_ids_field = fields['rewrite_target_ids']
            # 获取tokens
            rewrite_input_tokens = rewrite_input_ids_field.tokens
            rewrite_target_tokens = rewrite_target_ids_field.tokens
            for token in rewrite_input_tokens:
                # get the rewrite token id
                self._get_pretrained_token_id(token)
            for token in rewrite_target_tokens:
                self._get_pretrained_token_id(token)
            # 重新赋值新的token field
            rewrite_input_ids_field.tokens = rewrite_input_tokens
            rewrite_target_ids_field.tokens = rewrite_target_tokens
            fields['rewrite_input_ids'] = rewrite_input_ids_field
            fields['rewrite_target_ids'] = rewrite_target_ids_field
        return Instance(fields)

