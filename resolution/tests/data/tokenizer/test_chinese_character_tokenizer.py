# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-25
from unittest import TestCase, main

from resolution.common.data.tokenizer import ChineseCharacterTokenizer


class TestChineseCharacterTokenizer(TestCase):
    def setUp(self):
        super().setUp()
        self.word_tokenizer = ChineseCharacterTokenizer()

    def test_tokenize_all_chinese_tokens(self):
        sentence = "这是一个测试用例"
        expected_tokens = ["这", "是", "一", "个", "测", "试", "用", "例"]

        tokens = self.word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]

        assert token_text == expected_tokens

    def test_tokenize_chinese_and_english_tokens(self):
        sentence = "这是一个Test Case"
        expected_tokens = ["这", "是", "一", "个", "test", "case"]

        tokens = self.word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]

        assert token_text == expected_tokens


if __name__ == '__main__':
    main()