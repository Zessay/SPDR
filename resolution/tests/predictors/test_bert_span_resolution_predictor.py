# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-25
from unittest import TestCase, main
from allennlp.data import Vocabulary

from resolution.common.data.reader import BertSpanResolutionReader
from resolution.common.models import BertSpanPointerResolution
from resolution.common.predictors import BertSpanResolutionPredictor


class TestBertSpanResolutionPredictor(TestCase):
    def setUp(self):
        super().setUp()
        model_name = "adabrain/tests/data/base_bert/config.json"
        self.reader = BertSpanResolutionReader(model_name)
        self.vocab = Vocabulary()
        self.model = BertSpanPointerResolution(self.vocab, model_name)
        self.predictor = BertSpanResolutionPredictor(self.model, self.reader)

    def test_span_resolution_predictor(self):
        context = "能给我签名吗<EOS>出专辑再议"
        query = "我现在就要"

        result = self.predictor.predict(context, query)

        assert isinstance(result, dict)
        assert "rewrite_results" in result
        assert isinstance(result["rewrite_results"], str)


if __name__ == '__main__':
    main()