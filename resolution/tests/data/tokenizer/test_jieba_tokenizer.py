# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-13
from unittest import TestCase, main

from resolution.common.data.tokenizer import JiebaTokenizer


class TestJiebaTokenizer(TestCase):
    def test_tokenize_all_chinese_tokens(self):
        word_tokenizer = JiebaTokenizer()
        sentence = "这是一个测试用例"
        expected_tokens = ["这是", "一个", "测试用例"]

        tokens = word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]
        assert token_text == expected_tokens

    def test_tokenize_chinese_and_english_tokens(self):
        word_tokenizer = JiebaTokenizer()
        sentence = "这是一个Test Case"
        expected_tokens = ["这是", "一个", "test", " ", "case"]

        tokens = word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]
        assert token_text == expected_tokens

    def test_never_split_token(self):
        word_tokenizer = JiebaTokenizer()
        sentence = "我想了解一下ETC"
        expected_tokens = ["我", "想", "了解", "一下", "etc"]
        tokens = word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]
        assert token_text == expected_tokens

        word_tokenizer = JiebaTokenizer(never_split=["我想"])
        expected_tokens = ["我想", "了解", "一下", "etc"]
        tokens = word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]
        assert token_text == expected_tokens

    def test_add_start_and_end_tokens(self):
        word_tokenizer = JiebaTokenizer(start_tokens=["[CLS]"],
                                        end_tokens=["[SEP]"])
        sentence = "这是一个测试用例"
        expected_tokens = ["[CLS]", "这是", "一个", "测试用例", "[SEP]"]
        tokens = word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]
        assert token_text == expected_tokens

if __name__ == '__main__':
    main()