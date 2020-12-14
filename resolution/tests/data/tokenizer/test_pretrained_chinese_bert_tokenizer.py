# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-25
from unittest import TestCase, main
from resolution.common.data.tokenizer import PretrainedChineseBertTokenizer


class TestPretrainedChineseBertTokenizer(TestCase):
    def setUp(self) -> None:
        super().setUp()
        bert_model_path = "adabrain/tests/data/base_bert"
        self.word_tokenizer = PretrainedChineseBertTokenizer(model_name=bert_model_path)

    def test_splits_chinese(self):
        sentence = "这是一个测试用例"
        expected_tokens = ["[CLS]", "这", "是", "一", "个", "测", "试", "用", "例", "[SEP]"]

        tokens = self.word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]
        assert token_text == expected_tokens

    def test_splits_chinese_and_english(self):
        sentence = "这是一个AllenNLP测试用例"
        expected_tokens = ['[CLS]', '这', '是', '一', '个', 'allen', '##n', '##lp', '测', '试', '用', '例', '[SEP]']

        tokens = self.word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]
        assert token_text == expected_tokens


if __name__ == '__main__':
    main()