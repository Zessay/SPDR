# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-03
from typing import List, Optional, Tuple
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
from allennlp.data.token_indexers import PretrainedTransformerIndexer

from resolution.common.data.tokenizer.token import TokenAdd
from resolution.common.data.tokenizer.pretrained_chinese_bert_tokenizer \
    import PretrainedChineseBertTokenizer


@TokenIndexer.register("pretrained_chinese_bert")
class PretrainedChineseBertIndexer(PretrainedTransformerIndexer):
    """
    This `TokenIndexer` assumes that Tokens already have their indexes in them (see `text_id` field).
    We still require `model_name` because we want to form allennlp vocabulary from pretrained one.
    This `Indexer` is only really appropriate to use if you've also used a
    corresponding :class:`PretrainedBertTokenizer` to tokenize your input.  Otherwise you'll
    have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

    Registered as a `TokenIndexer` with name "pretrained_chinese_bert".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = `None`)
        If not None, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedBertEmbedder`.
    """

    def __init__(self,
                 model_name: str,
                 namespace: str = "tags",
                 max_length: int = None,
                 **kwargs) -> None:
        TokenIndexer.__init__(self, **kwargs)
        self._namespace = namespace
        self._allennlp_tokenizer = PretrainedChineseBertTokenizer(model_name)
        self._tokenizer = self._allennlp_tokenizer.tokenizer
        self._added_to_vocabulary = False

        self._num_added_start_tokens = len(self._allennlp_tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(self._allennlp_tokenizer.single_sequence_end_tokens)

        self._max_length = max_length
        if self._max_length is not None:
            num_added_tokens = len(self._allennlp_tokenizer.tokenize("a")) - 1
            self._effective_max_length = (  # we need to take into account special tokens
                self._max_length - num_added_tokens
            )
            if self._effective_max_length <= 0:
                raise ValueError(
                    "max_length needs to be greater than the number of special tokens inserted."
                )

    @overrides
    def tokens_to_indices(self,
                          tokens: List[TokenAdd],
                          vocabulary: Vocabulary) -> IndexedTokenList:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)

        indices, type_ids, turn_ids = self._extract_token_and_type_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        output: IndexedTokenList = {
            "token_ids": indices,
            "mask": [True] * len(indices),
            "type_ids": type_ids,
            "turn_ids": turn_ids
        }

        return self._postprocess_output(output)

    def _extract_token_and_type_ids(self, tokens: List[TokenAdd]
                                    ) -> Tuple[List[int], Optional[List[int]], Optional[List[int]]]:
        """
        Roughly equivalent to `zip(*[(token.text_id, token.type_id) for token in tokens])`,
        with some checks.
        """
        indices: List[int] = []
        type_ids: List[int] = []
        turn_ids: List[int] = []
        for token in tokens:
            if getattr(token, "text_id", None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead. Id comes from the pretrained vocab.
                # It is computed in PretrainedBertTokenizer.
                indices.append(token.text_id)
            else:
                raise KeyError(
                    "Using PretrainedChineseBertIndexer but field text_id is not set"
                    f" for the following token: {token.text}"
                )

            if type_ids is not None and getattr(token, "type_id", None) is not None:
                type_ids.append(token.type_id)
            else:
                type_ids.append(0)

            if getattr(token, "turn_id", None) is not None:
                turn_ids.append(token.turn_id)
            else:
                turn_ids.append(0)

        return indices, type_ids, turn_ids

    def __eq__(self, other):
        if isinstance(other, PretrainedChineseBertIndexer):
            for key in self.__dict__:
                if key == "_tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented
