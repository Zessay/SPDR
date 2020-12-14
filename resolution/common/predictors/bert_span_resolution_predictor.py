# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-15
import re
import os
import json
from overrides import overrides

from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive


@Predictor.register("bert_span_resolution")
class BertSpanResolutionPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader) -> None:
        super(BertSpanResolutionPredictor, self).__init__(model, dataset_reader)
        # 用于清洗context和query的pattern
        self._pattern = re.compile(r"\s+")

    def predict(self, context: str, query: str) -> JsonDict:
        # 去除context和query中的无效空格字符
        context = self._pattern.sub("", context)
        query = self._pattern.sub("", query)
        return self.predict_json({"context": context, "query": query})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        context, query, _ = self._dataset_reader.make_input_to_instance(
            json_dict['context'], json_dict['query'])
        return self._dataset_reader.text_to_instance(context, query)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        # 内部会调用`_json_to_instances`
        # 和Model中的 `forward_on_instances`
        results = self.predict_batch_json([inputs])
        assert len(results) == 1
        return results[0]


def load_model(model_path: str, predictor_name: str, device: int = -1):
    model_config = "bert_config.json"
    files = os.listdir(model_path)
    for file in files:
        if file.endswith("config.json"):
            model_config = file

    config_override = {
        "dataset_reader.model_name": model_path,
        "model.model_name": os.path.join(model_path, model_config),
        "model.task_pretrained_file": None
    }
    archive = load_archive(os.path.join(model_path, "model.tar.gz"),
                           cuda_device=device, overrides=json.dumps(config_override))
    predictor = Predictor.from_archive(archive, predictor_name)
    return predictor