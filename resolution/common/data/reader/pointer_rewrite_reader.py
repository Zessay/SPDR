# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-06
import itertools
from typing import Dict, List, Iterator, Optional
from overrides import overrides
import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, MetadataField
from allennlp.data.fields import ArrayField, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, Token
from allennlp.data import Vocabulary
from allennlp.common.util import JsonDict

from resolution.common.data.tokenizer import TokenAdd, ChineseCharacterTokenizer


@DatasetReader.register('pointer_rewrite')
class PointerRewriteReader(DatasetReader):
    """
    The dataset reader which is used for pointer rewrite network.

    # Parameters
    vocab: `Vocabulary`, optional (default = `None`)
    vocab_path : `str`, optional (default = `None`)
        The vocabulary path saved by `Vocabulary` class.
        Must specify one from `vocab` and `vocab_path`.
    max_dec_len : `int`, optional (default = `30`)
        The max decode length in the decode phrase.
    max_turn_len : `int`, optional (default = `3`)
        The max turn length od the dialogue (include context and query).
    index_name : `int`, optional (default = `tokens`)
        The namespace of vocabulary will be used when indexing.
    start_token : `str`, optional (default = `[CLS]`)
        The start token represents the beginning of a sentence.
    end_token : `str`, optional (default = `[SEP]`)
        The end token represents the end of a sentence.
    pad_token : `str`, optional (default = `[PAD]`)
        The pad token of the sentence.
    oov_token : `str`, optinal (default = `[UNK]`)
        The oov token of the sentence.
    do_lower_case : `bool`, optional (default = `False`)
        Whether do lower case to the word when tokenize.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default = `None`)
        How to convert a word to index, default use `SingleIdTokenIndexer`.
    tokenizer : `Tokenizer`, optional (default = `None`)
        How to tokenize a sentence, default chinese characters and punctuations
        will be split, and other words will save connected.
    lazy : `bool`, optional (default = `False`)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    """
    def __init__(self,
                 vocab: Vocabulary = None,
                 vocab_path: str = None,
                 max_enc_len: int = 512,
                 max_dec_len: int = 30,
                 max_turn_len: int = 3,
                 index_name: str = "tokens",
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
        super(PointerRewriteReader, self).__init__(lazy, **kwargs)
        if never_split is not None:
            never_split = [start_token, end_token, pad_token, oov_token] + never_split
        else:
            never_split = [start_token, end_token, pad_token, oov_token]
        # the max length of the input
        self.max_length = max_enc_len
        # Tokens
        self._start_token = start_token
        self._end_token = end_token
        self._max_dec_len = max_dec_len
        self._max_turn_len = max_turn_len - 1
        self._index_name = index_name
        if tokenizer is None:
            self._tokenizer = ChineseCharacterTokenizer(do_lower_case, never_split)
        else:
            self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {
            index_name: SingleIdTokenIndexer(namespace=index_name)}
        if vocab is None:
            self.vocab = Vocabulary.from_files(vocab_path, padding_token=pad_token, oov_token=oov_token)
        else:
            self.vocab = vocab
        # get the vocabulary_to_index and oov id
        self._token2index = self.vocab.get_token_to_index_vocabulary(namespace=index_name)
        self._unk_id = self.vocab.get_token_index(self.vocab._oov_token, namespace=index_name)
        self._vocab_size = self.vocab.get_vocab_size(namespace=index_name)

    def _get_turn_field_value(self,
                              context: List[List[Token]],
                              query: List[Token]):
        """Get the turn index of each token in context and query."""
        origin_turns = len(context)
        # total length of context
        total_len = sum([len(sent) for sent in context])
        context_turn = []

        if origin_turns > self._max_turn_len:
            # traverse from right to left
            cur_turn = self._max_turn_len - 1
            for sent in reversed(context):
                context_turn = [cur_turn] * len(sent) + context_turn
                cur_turn -= 1
                if cur_turn < 0:
                    break
            context_turn = [0] * (total_len - len(context_turn)) + context_turn
            query_turn = [self._max_turn_len] * len(query)
            context_turn_field = ArrayField(
                array=np.array(context_turn, dtype=np.int32),
                padding_value=self._max_turn_len - 1)
            query_turn_field = ArrayField(
                array=np.array(query_turn, dtype=np.int32),
                padding_value=self._max_turn_len)
        else:
            # traverse from left to right
            cur_turn = 0
            for sent in context:
                context_turn += [cur_turn] * len(sent)
                cur_turn += 1
            query_turn = [cur_turn] * len(query)
            context_turn_field = ArrayField(
                array=np.array(context_turn, dtype=np.int32),
                padding_value=cur_turn - 1)
            query_turn_field = ArrayField(
                array=np.array(query_turn, dtype=np.int32),
                padding_value=cur_turn)
        assert len(context_turn) == total_len
        return context_turn_field, query_turn_field

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

    def get_common_field(self, context_flat: List[TokenAdd], query: List[TokenAdd],
                         rewrite: Optional[List[TokenAdd]] = None):
        fields: Dict[str, Field] = {}
        # inspect the oov words in the context and query
        # and get the extend ids with oov words
        extend_context_ids, oovs = self.context2ids(context_words=context_flat)
        extend_query_ids, oovs = self.query2ids(query_words=query, oovs=oovs)
        oovs_len = LabelField(label=len(oovs), label_namespace="len_tags",
                              skip_indexing=True)
        context_len_field = LabelField(label=len(context_flat),
                                       label_namespace="len_tags",
                                       skip_indexing=True)
        query_len_field = LabelField(label=len(query),
                                     label_namespace="len_tags",
                                     skip_indexing=True)
        fields['extend_context_ids'] = ArrayField(
            np.array(extend_context_ids, dtype=np.int32))
        fields['extend_query_ids'] = ArrayField(
            np.array(extend_query_ids, dtype=np.int32))
        # preserve the length info in order to get the mask
        fields['oovs_len'] = oovs_len
        fields['context_len'] = context_len_field
        fields['query_len'] = query_len_field
        # preserve the original text
        metadata = {
            "context_words": "".join([token.text for token in context_flat[1:-1]]),  # str
            "query_words": [token.text for token in query][:-1],  # List[str]
            "oovs": oovs         # List[str]
        }

        if rewrite is not None:
            rewrite_input_tokens, rewrite_targ_tokens = self.get_dec_inp_targ_seqs(rewrite)
            # get the extend rewrite ids
            extend_rewrite_ids = self.rewrite2ids(rewrite_words=rewrite_targ_tokens,
                                                  oovs=oovs)
            rewrite_len_field = LabelField(label=len(rewrite_input_tokens),
                                           label_namespace="len_tags",
                                           skip_indexing=True)

            rewrite_input_tokens_field = TextField(rewrite_input_tokens,
                                                   self._token_indexers)
            rewrite_targ_tokens_field = TextField(rewrite_targ_tokens,
                                                  self._token_indexers)

            fields['rewrite_input_ids'] = rewrite_input_tokens_field
            fields['rewrite_target_ids'] = rewrite_targ_tokens_field
            fields['extend_rewrite_ids'] = ArrayField(
                np.array(extend_rewrite_ids, dtype=np.int32))
            fields['rewrite_len'] = rewrite_len_field
            metadata["rewrite"] = [token.text for token in rewrite]  # List[str]

        fields['metadata'] = MetadataField(metadata)
        return fields

    @overrides
    def text_to_instance(self,
                         context: List[List[TokenAdd]],
                         query: List[TokenAdd],
                         rewrite: Optional[List[TokenAdd]] = None) -> Instance:
        """
        Convert `context`, `query` and `rewrite` to `Instance`, comprised of `Field`.
        :param context: `List[List[TokenAdd]]`, include multiple turns, each turn consists of `Token`.
        :param query: `List[TokenAdd]`, consists of tokenized `Token` of query.
        :param rewrite: `List[TokenAdd]`, optional.
        :return: `Instance`, include various of `Field`.
        """
        # flatten context
        context_flat = list(itertools.chain(*context))

        # get the common fields
        fields = self.get_common_field(context_flat, query, rewrite)
        # get the turn field of context and query
        context_turn_field, query_turn_field = self._get_turn_field_value(context, query)

        context_tokens_field = TextField(context_flat, self._token_indexers)
        query_tokens_field = TextField(query, self._token_indexers)

        fields['context_ids'] = context_tokens_field
        fields['query_ids'] = query_tokens_field

        fields['context_turn'] = context_turn_field
        fields['query_turn'] = query_turn_field
        return Instance(fields)

    def get_dec_inp_targ_seqs(self, rewrite_words: List[Token]):
        """
        Add `START` to the beginning of target input sequence,
        while add `END` to the end of target output sequence.
        """
        inp = [TokenAdd(text=self._start_token)] + rewrite_words[:]
        targ = rewrite_words[:]

        if len(inp) > self._max_dec_len:
            inp = inp[:self._max_dec_len]
            targ = targ[:self._max_dec_len]
        else:
            targ.append(TokenAdd(text=self._end_token))
        assert len(inp) == len(targ)

        return inp, targ

    def context2ids(self, context_words: List[Token]):
        """
        Get the extend ids of context, mainly extend `unk` positions,
        the first `unk` will be convert to `unk1`, the second will be
        `unk2`, etc.
        """
        extend_context_ids = []  # save the extend ids of context
        oovs = []  # List[str], save the oov words

        for token in context_words:
            word = token.text
            # if the word is oov words
            if word not in self._token2index:
                # if the word not in our oov list
                if word not in oovs:
                    oovs.append(word)
                oov_num = oovs.index(word)
                extend_context_ids.append(
                    self._vocab_size + oov_num - self._unk_id)
            else:
                extend_context_ids.append(0)
        return extend_context_ids, oovs

    def query2ids(self, query_words: List[Token], oovs: List[str]):
        """
        The same as `context2ids`, and continue to use the last `unk` index.
        """
        extend_query_ids = []

        for token in query_words:
            word = token.text
            if word not in self._token2index:
                if word not in oovs:
                    oovs.append(word)
                oov_num = oovs.index(word)
                extend_query_ids.append(
                    self._vocab_size + oov_num - self._unk_id)
            else:
                extend_query_ids.append(0)
        return extend_query_ids, oovs

    def rewrite2ids(self, rewrite_words: List[Token], oovs: List[str]):
        """
        Use the context and query `unk` ids to extend the rewrite target output
        sequence (the target input remain unchanged.)
        """
        extend_rewrite_ids = []

        for token in rewrite_words:
            word = token.text
            if word not in self._token2index:
                if word in oovs:
                    vocab_idx = self._vocab_size + oovs.index(word) - self._unk_id
                    extend_rewrite_ids.append(vocab_idx)
                else:
                    extend_rewrite_ids.append(0)
            else:
                extend_rewrite_ids.append(0)
        return extend_rewrite_ids

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        Read the train samples in a file and yield `Instance`.
        :param file_path: `str`, the path of the file
            Each line of the file is like:
            `context_1<EOS>context_2<EOS>...`\t\t`query`\t\t`rewrite`

            Before the first `\t\t` is the context, separate each turn with `<EOS>`.
        :return:
        """
        with open(file_path, 'r', encoding="utf-8") as f:
            for line in f:
                # position 0 is context, position 1 is query
                # position 2 is rewrite
                line_list = line.strip().split("\t\t")
                if len(line_list) > 3:
                    context, query, rewrite, *_ = line_list
                else:
                    context, query, rewrite = line_list
                context, query = self.make_input_to_instance(context, query)

                # get rewrite tokens
                rewrite = self._tokenizer.tokenize(rewrite)

                yield self.text_to_instance(context=context, query=query, rewrite=rewrite)

    def make_input_to_instance(self, context: str, query: str):
        contexts = context.split("<EOS>")
        # 分词
        context = [self._tokenizer.tokenize(sent) for sent in contexts]
        query = self._tokenizer.tokenize(query)
        # 添加special tokens
        for i in range(len(context)):
            if i == 0:
                context[i] = [TokenAdd(text=self._start_token)] + context[i] + [TokenAdd(text=self._end_token)]
            else:
                context[i] = context[i] + [TokenAdd(text=self._end_token)]

        # 在query前后加上special token
        query = query + [TokenAdd(text=self._end_token)]
        # limit the max length of context
        context, _ =  self._limit_max_length(context, query)
        return context, query

    def text_to_prediction_instance(self, json_dict: JsonDict):
        """
        The predictor will use this method.
        :param json_dict: `JsonDict`, required
            The `context` and `query` must be in the dict keys,
            and the `rewrite` is optional.
            The different turns of `context` is separate with `<EOS>`.
        """
        context = json_dict['context']
        query = json_dict['query']

        # context: START + context_1 + END + context_2 + END
        # query: query + END
        context, query = self.make_input_to_instance(context, query)

        rewrite = None
        if 'rewrite' in json_dict and json_dict['rewrite'] is not None and json_dict['rewrite'] != '':
            rewrite_string = json_dict['rewrite']
            rewrite = self._tokenizer.tokenize(rewrite_string)

        return self.text_to_instance(context=context, query=query, rewrite=rewrite)
