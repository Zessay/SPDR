# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-25
from pathlib import Path
from allennlp.common.testing import ModelTestCase

PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"


class TestBertSpanPointerResolution(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            param_file=FIXTURES_ROOT / "span_pointer_resolution" / "bert_span_pointer_resolution.jsonnet",
            dataset_file=FIXTURES_ROOT / "test_pointer_rewrite.txt"
        )

    def test_simple_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            gradients_to_ignore={"_text_field_embedder._matched_embedder.transformer_model.pooler.dense.weight",
                                 "_text_field_embedder._matched_embedder.transformer_model.pooler.dense.bias"})
