# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-11
import json
import tempfile
from pathlib import Path
from unittest import TestCase, main

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common import Params
from allennlp.models import Model

from resolution.common.predictors import PointerRewritePredictor

PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"


class TestPointerRewritePredictor(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.context = "能给我签名吗<EOS>出专辑再议"
        self.query = "我现在就要"

    def test_lstm_lstm_predictor(self):
        param_file = FIXTURES_ROOT / "pointer_rewrite" / "lstm_lstm_pointer_rewrite.jsonnet"
        params = Params.from_file(param_file)
        # 获取reader
        reader = DatasetReader.from_params(params["dataset_reader"])
        # 获取模型
        # 如果存在词表的参数，则加载词表
        if "vocabulary" in params:
            vocab_params = params["vocabulary"]
            vocab = Vocabulary.from_params(params=vocab_params)
        else:
            vocab = Vocabulary()

        # 加载模型
        model = Model.from_params(params=params["model"], vocab=vocab)

        predictor = PointerRewritePredictor(dataset_reader=reader, model=model)
        result = predictor.predict(self.context, self.query)

        self.assertTrue("rewrite_results" in result)
        assert isinstance(result["rewrite_results"], str)

    def test_bert_transformer_predictor(self):
        param_file = FIXTURES_ROOT / "pointer_rewrite" / "bert_transformer_pointer_rewrite.jsonnet"
        params = Params.from_file(param_file)
        # 构建适用于bert model的词表，和vocabulary词表保持一致
        vocab_path = params["dataset_reader"]["vocab_path"]
        # 新生成的bert词表的路径
        bert_temp_dir = tempfile.mkdtemp(suffix="bert")
        with open(Path(vocab_path) / "tokens.txt", 'r', encoding="utf-8") as f, \
            open(Path(bert_temp_dir) / "vocab.txt", 'w', encoding="utf-8") as fp:
            fp.write("[PAD]"+"\n")
            for line in f:
                line = line.strip()
                fp.write(line)
                fp.write("\n")

        # 改写config中的部分参数
        overrides_config = {
            "dataset_reader.model_name": bert_temp_dir,
            "model.model_name": params["model"]["model_name"] + "/config.json"
        }
        overrides_config = json.dumps(overrides_config)
        # 重新加载参数并重写其中部分参数
        params = Params.from_file(param_file, params_overrides=overrides_config)

        # 获取reader
        reader = DatasetReader.from_params(params["dataset_reader"])
        # 如果存在词表的参数，则加载词表
        if "vocabulary" in params:
            vocab_params = params["vocabulary"]
            vocab = Vocabulary.from_params(params=vocab_params)
        else:
            vocab = Vocabulary()
        # 加载模型
        # 将模型对应的model_name改成对应的config文件
        model = Model.from_params(params=params["model"], vocab=vocab)

        predictor = PointerRewritePredictor(dataset_reader=reader, model=model)
        result = predictor.predict(self.context, self.query)

        self.assertTrue("rewrite_results" in result)
        assert isinstance(result["rewrite_results"], str)


if __name__ == '__main__':
    main()