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

    def _adjust_label(self, mask_string: str, start_string: str, end_string: str, sub_len: int):
        if mask_string and start_string and end_string:
            # 这里由于去除了query前面的[CLS]标志，所以自动向后顺延一位
            # 即预测对应位置前面的span情况
            mask_label = [int(m) for m in mask_string.split(',')][:-1]
            start_label = [int(s) - sub_len for s in start_string.split(',')][:-1]
            end_label = [int(e) - sub_len for e in end_string.split(',')][:-1]

            # 将-1转化为0
            start_label = [max(0, s) for s in start_label]
            end_label = [max(0, e) for e in end_label]
        else:
            mask_label, start_label, end_label = None, None, None
        return mask_label, start_label, end_label

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        context, query, sub_len = self._dataset_reader.make_input_to_instance(
            json_dict['context'], json_dict['query'])
        rewrite = None if "rewrite" not in json_dict else json_dict["rewrite"]
        # 对rewrite进行分词
        rewrite = [token.text for token in self._dataset_reader._tokenizer.tokenize(rewrite)]
        mask_string = None if "mask_string" not in json_dict else json_dict["mask_string"]
        start_string = None if "start_string" not in json_dict else json_dict["start_string"]
        end_string = None  if "end_string" not in json_dict else json_dict["end_string"]
        restore_string = None if "restore_string" not in json_dict else json_dict["restore_string"]

        # 如果由于句子过长删除了前面的一些语句，则需要将start和end的标签进行矫正
        mask_label, start_label, end_label = self._adjust_label(mask_string,
                                                                start_string,
                                                                end_string, sub_len)

        return self._dataset_reader.text_to_instance(context, query,
                                                     rewrite=rewrite,
                                                     mask_label=mask_label,
                                                     start_label=start_label,
                                                     end_label=end_label,
                                                     restore_string=restore_string)

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