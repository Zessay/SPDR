# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-11
from pathlib import Path
from unittest import TestCase, main

from resolution.common.data.reader import PointerRewriteReader

PROJECT_ROOT = (Path(__file__).parent / ".." / ".." / "..").resolve()
FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"


class TestPointerRewriteReader(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # The vocab file saved with allennlp Vocabulary
        vocab_path = "adabrain/tests/data/vocabulary"
        self.data_reader = PointerRewriteReader(vocab_path=vocab_path,
                                                oov_token="@@UNKNOWN@@")

    def test_pointer_rewrite_reader(self):
        train_dataset_path = FIXTURES_ROOT / "test_pointer_rewrite.txt"
        instance = list(self.data_reader.read(train_dataset_path))

        self.assertEqual(len(instance), 2, "data length error")
        self.assertEqual(len(instance[0]), 14, "instance length error")

        context_words = instance[0]['metadata']['context_words']
        query_words = instance[0]['metadata']['query_words']
        rewrite_words = instance[0]['metadata']['rewrite']

        self.assertEqual(context_words, "能给我签名吗[SEP]出专辑再议")
        self.assertEqual(query_words, ['我', '现', '在', '就', '要'])
        self.assertEqual(rewrite_words, ['我', '现', '在', '就', '要', '签', '名'])

if __name__ == '__main__':
    main()
