# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-25
from pathlib import Path
from unittest import TestCase, main

from resolution.common.data.reader import BertSpanResolutionReader

PROJECT_ROOT = (Path(__file__).parent / ".." / ".." / "..").resolve()
FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"


class TestBertSpanResolutionReader(TestCase):
    def setUp(self):
        super().setUp()
        bert_model_path = "adabrain/tests/data/base_bert"
        self.data_reader = BertSpanResolutionReader(model_name=bert_model_path)

    def test_bert_span_resolution_reader(self):
        train_dataset_path = FIXTURES_ROOT / "test_pointer_rewrite.txt"

        instance = self.data_reader.read(train_dataset_path)

        assert len(instance) == 2
        assert len(instance[0]) == 8

        context_tokens = instance[0]["metadata"]["context_tokens"]
        query_tokens = instance[0]["metadata"]["query_tokens"]

        assert context_tokens == ["[CLS]", "能", "给", "我", "签", "名", "吗", "[SEP]",
                                  "出", "专", "辑", "再", "议", "[SEP]"]
        assert query_tokens == ["我", "现", "在", "就", "要", "[SEP]"]


if __name__ == "__main__":
    main()