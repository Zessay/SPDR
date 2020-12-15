# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-02
import copy
from typing import Any, Dict, List, Optional, Tuple

from overrides import overrides
from transformers import AutoTokenizer
from transformers import BertTokenizer, ElectraTokenizer

from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from resolution.common.data.tokenizer.token import TokenAdd


@Tokenizer.register("pretrained_chinese_bert")
class PretrainedChineseBertTokenizer(PretrainedTransformerTokenizer):
    def __init__(self,
                 model_name: str,
                 add_special_tokens: bool = True,
                 max_length: Optional[int] = None,
                 stride: int = 0,
                 truncation_strategy: str = "longest_first",
                 tokenizer_kwargs: Optional[Dict[str, Any]] = None,
                 ) -> None:
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        else:
            tokenizer_kwargs = tokenizer_kwargs.copy()
        tokenizer_kwargs.setdefault("use_fast", True)
        # Note: Just because we request a fast tokenizer doesn't mean we get
        # one.

        model_name = model_name.rstrip("/")
        postfix = model_name.split("/")[-1]
        if postfix.endswith("config.json"):
            model_name = "/".join(model_name.split("/")[:-1])
            postfix = model_name.split("/")[-1]
        if "bert" in postfix or "chinese" in postfix:
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name, add_special_tokens=False, **tokenizer_kwargs)
        elif postfix.startswith("electra"):
            self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._add_special_tokens = add_special_tokens
        self._max_length = max_length
        self._stride = stride
        self._truncation_strategy = truncation_strategy

        self._tokenizer_lowercases = self.tokenizer_lowercases(self.tokenizer)

        try:
            self._reverse_engineer_special_tokens(
                "a", "b", model_name, tokenizer_kwargs)
        except AssertionError:
            # For most transformer models, "a" and "b" work just fine as dummy tokens.  For a few,
            # they don't, and so we use "1" and "2" instead.
            self._reverse_engineer_special_tokens(
                "1", "2", model_name, tokenizer_kwargs)

    def _reverse_engineer_special_tokens(self,
                                         token_a: str,
                                         token_b: str,
                                         model_name: str,
                                         tokenizer_kwargs: Optional[Dict[str, Any]],):
        # storing the special tokens
        self.sequence_pair_start_tokens = []
        self.sequence_pair_mid_tokens = []
        self.sequence_pair_end_tokens = []
        # storing token type ids for the sequences
        self.sequence_pair_first_token_type_id = None
        self.sequence_pair_second_token_type_id = None

        # storing the special tokens
        self.single_sequence_start_tokens = []
        self.single_sequence_end_tokens = []
        # storing token type id for the sequence
        self.single_sequence_token_type_id = None

        # Reverse-engineer the tokenizer for two sequences
        postfix = model_name.split("/")[-1]
        if postfix.endswith("config.json"):
            model_name = "/".join(model_name.split("/")[:-1])
            postfix = model_name.split("/")[-1]
        if "bert" in postfix or "chinese" in postfix:
            tokenizer_with_special_tokens = BertTokenizer.from_pretrained(
                model_name, add_special_tokens=False, **tokenizer_kwargs)
        elif postfix.startswith("electra"):
            tokenizer_with_special_tokens = ElectraTokenizer.from_pretrained(model_name)
        else:
            tokenizer_with_special_tokens = AutoTokenizer.from_pretrained(model_name)

        dummy_output = tokenizer_with_special_tokens.encode_plus(
            token_a,
            token_b,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=False,
        )
        dummy_a = self.tokenizer.encode(
            token_a,
            add_special_tokens=False,
            add_prefix_space=True)[0]
        assert dummy_a in dummy_output["input_ids"]
        dummy_b = self.tokenizer.encode(
            token_b,
            add_special_tokens=False,
            add_prefix_space=True)[0]
        assert dummy_b in dummy_output["input_ids"]
        assert dummy_a != dummy_b

        seen_dummy_a = False
        seen_dummy_b = False
        for token_id, token_type_id in zip(dummy_output["input_ids"],
                                           dummy_output["token_type_ids"]):
            if token_id == dummy_a:
                if seen_dummy_a or seen_dummy_b:  # seeing a twice or b before a
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_a = True
                assert (
                    self.sequence_pair_first_token_type_id is None
                    or self.sequence_pair_first_token_type_id == token_type_id
                ), "multiple different token type ids found for the first sequence"
                self.sequence_pair_first_token_type_id = token_type_id
                continue

            if token_id == dummy_b:
                if seen_dummy_b:  # seeing b twice
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_b = True
                assert (
                    self.sequence_pair_second_token_type_id is None
                    or self.sequence_pair_second_token_type_id == token_type_id
                ), "multiple different token type ids found for the second sequence"
                self.sequence_pair_second_token_type_id = token_type_id
                continue

            token = TokenAdd(
                tokenizer_with_special_tokens.convert_ids_to_tokens(token_id),
                text_id=token_id,
                type_id=token_type_id,
            )
            if not seen_dummy_a:
                self.sequence_pair_start_tokens.append(token)
            elif not seen_dummy_b:
                self.sequence_pair_mid_tokens.append(token)
            else:
                self.sequence_pair_end_tokens.append(token)

        assert (
            len(self.sequence_pair_start_tokens)
            + len(self.sequence_pair_mid_tokens)
            + len(self.sequence_pair_end_tokens)
        ) == self.tokenizer.num_special_tokens_to_add(pair=True)

        # Reverse-engineer the tokenizer for one sequence
        dummy_output = tokenizer_with_special_tokens.encode_plus(
            token_a,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=False,
            add_prefix_space=True,
        )

        seen_dummy_a = False
        for token_id, token_type_id in zip(dummy_output["input_ids"],
                                           dummy_output["token_type_ids"]):
            if token_id == dummy_a:
                if seen_dummy_a:
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_a = True
                assert (
                    self.single_sequence_token_type_id is None
                    or self.single_sequence_token_type_id == token_type_id
                ), "multiple different token type ids found for the sequence"
                self.single_sequence_token_type_id = token_type_id
                continue

            token = TokenAdd(
                tokenizer_with_special_tokens.convert_ids_to_tokens(token_id),
                text_id=token_id,
                type_id=token_type_id,
            )
            if not seen_dummy_a:
                self.single_sequence_start_tokens.append(token)
            else:
                self.single_sequence_end_tokens.append(token)

        assert (
            len(self.single_sequence_start_tokens) + len(self.single_sequence_end_tokens)
        ) == self.tokenizer.num_special_tokens_to_add(pair=False)

    @overrides
    def tokenize(self, text: str) -> List[TokenAdd]:
        """
        This method only handles a single sentence (or sequence) of text.
        """
        encoded_tokens = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=False,
            max_length=self._max_length,
            stride=self._stride,
            truncation_strategy=self._truncation_strategy,
            return_tensors=None,
            return_offsets_mapping=self.tokenizer.is_fast,
            return_attention_mask=False,
            return_token_type_ids=True,
        )
        # token_ids contains a final list with ids for both regular and special
        # tokens
        token_ids, token_type_ids, token_offsets = (
            encoded_tokens["input_ids"],
            encoded_tokens["token_type_ids"],
            encoded_tokens.get("offset_mapping"),
        )

        # If we don't have token offsets, try to calculate them ourselves.
        if token_offsets is None:
            token_offsets = self._estimate_character_indices(text, token_ids)

        tokens = []
        for token_id, token_type_id, offsets in zip(
                token_ids, token_type_ids, token_offsets):
            if offsets is None or offsets[0] >= offsets[1]:
                start = None
                end = None
            else:
                start, end = offsets

            tokens.append(
                TokenAdd(
                    text=self.tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=False),
                    text_id=token_id,
                    type_id=token_type_id,
                    idx=start,
                    idx_end=end,
                ))

        if self._add_special_tokens:
            tokens = self.add_special_tokens(tokens)

        return tokens

    def _intra_word_tokenize(
        self, string_tokens: List[TokenAdd]
    ) -> Tuple[List[TokenAdd], List[Optional[Tuple[int, int]]]]:
        tokens: List[TokenAdd] = []
        offsets: List[Optional[Tuple[int, int]]] = []
        for token in string_tokens:
            token_string = token.text
            wordpieces = self.tokenizer.encode_plus(
                token_string,
                add_special_tokens=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            wp_ids = wordpieces["input_ids"]
            # add other attributes of the token to the wordpieces
            attrs = dir(token)   # get all of the attributes of the token
            attrs_dict = {}  # save the attribute of the token
            for attr in attrs:
                if attr.startswith("__"):
                    continue
                elif attr in ['text', 'text_id']:
                    continue
                else:
                    attrs_dict[attr] = getattr(
                        token, attr)  # get the value of the attribute

            if len(wp_ids) > 0:
                offsets.append((len(tokens), len(tokens) + len(wp_ids) - 1))
                tokens.extend(
                    TokenAdd(text=wp_text, text_id=wp_id, **attrs_dict)
                    for wp_id, wp_text in zip(wp_ids, self.tokenizer.convert_ids_to_tokens(wp_ids))
                )
            else:
                offsets.append(None)
        return tokens, offsets

    def intra_word_tokenize(
        self, string_tokens: List[TokenAdd], add_special_tokens: bool = True
    ) -> Tuple[List[TokenAdd], List[Optional[Tuple[int, int]]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that tokens[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.

        This function inserts special tokens.
        """
        tokens, offsets = self._intra_word_tokenize(string_tokens)
        # Whether to add special tokens
        if add_special_tokens:
            tokens = self.add_special_tokens(tokens)
            offsets = self._increment_offsets(
                offsets, len(self.single_sequence_start_tokens))
        return tokens, offsets

    def intra_word_tokenize_sentence_pair(
        self, string_tokens_a: List[TokenAdd], string_tokens_b: List[TokenAdd],
        add_special_tokens: bool = True
    ) -> Tuple[List[TokenAdd], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that wordpieces[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.

        This function inserts special tokens.
        """
        tokens_a, offsets_a = self._intra_word_tokenize(string_tokens_a)
        tokens_b, offsets_b = self._intra_word_tokenize(string_tokens_b)
        offsets_b = self._increment_offsets(
            offsets_b,
            (
                len(self.sequence_pair_start_tokens)
                + len(tokens_a)
                + len(self.sequence_pair_mid_tokens)
            ),
        )
        if add_special_tokens:
            tokens_a = self.add_special_tokens(tokens_a, tokens_b)
            offsets_a = self._increment_offsets(
                offsets_a, len(self.sequence_pair_start_tokens))

        return tokens_a, offsets_a, offsets_b

    def add_special_tokens(
        self, tokens1: List[TokenAdd], tokens2: Optional[List[TokenAdd]] = None
    ) -> List[TokenAdd]:
        # Make sure we don't change the input parameters
        tokens1 = copy.deepcopy(tokens1)
        tokens2 = copy.deepcopy(tokens2)

        # We add special tokens and also set token type ids.
        if tokens2 is None:
            for token in tokens1:
                token.type_id = self.single_sequence_token_type_id
            return self.single_sequence_start_tokens + \
                tokens1 + self.single_sequence_end_tokens
        else:
            for token in tokens1:
                token.type_id = self.sequence_pair_first_token_type_id
            for token in tokens2:
                token.type_id = self.sequence_pair_second_token_type_id
            return (
                self.sequence_pair_start_tokens
                + tokens1
                + self.sequence_pair_mid_tokens
                + tokens2
                + self.sequence_pair_end_tokens
            )