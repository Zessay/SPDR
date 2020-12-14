# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-25
from unittest import TestCase, main
from allennlp.data.vocabulary import Vocabulary

from resolution.common.data.token_indexer import PretrainedChineseBertIndexer
from resolution.common.data.tokenizer.pretrained_chinese_bert_tokenizer \
    import PretrainedChineseBertTokenizer, BertTokenizer


class TestPretrainedChineseBertIndexer(TestCase):
    def test_as_array_produces_token_sequence_bert(self):
        bert_model_path = "adabrain/tests/data/base_bert"
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_model_path)
        allennlp_tokenizer = PretrainedChineseBertTokenizer(model_name=bert_model_path, add_special_tokens=False)
        indexer = PretrainedChineseBertIndexer(model_name=bert_model_path)
        string_specials = "[CLS]这是一个测试用例[SEP]"

        tokens = tokenizer.tokenize(string_specials)
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)

        # allennlp indexer
        allennlp_tokens = allennlp_tokenizer.tokenize(string_specials)
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens, vocab)
        assert indexed["token_ids"] == expected_ids
        assert "turn_ids" in indexed


if __name__ == "__main__":
    main()