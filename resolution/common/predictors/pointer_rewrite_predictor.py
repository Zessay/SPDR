# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-11
import os
import json
import torch
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive

@Predictor.register("pointer_for_rewrite")
class PointerRewritePredictor(Predictor):
    def predict(self, context: str, query: str,
                rewrite: str = None) -> JsonDict:
        """
        Do rewrite according to context and current query.
        :param context: str, the history dialogue, <EOS> token between the different turn.
        :param query: str, the current query.
        :return:
        """
        with torch.no_grad():
            return self.predict_json({"context": context,
                                      "query": query,
                                      "rewrite": rewrite})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_prediction_instance(json_dict)


def load_model(vocab_path: str, model_path: str, predictor_name: str, device: int = -1):
    model_config = None
    files = os.listdir(model_path)
    for file in files:
        if file.endswith("config.json"):
            model_config = file

    # 如果model_config是None
    # 说明是LSTM和Transformer的Encoder-Decoder
    # 没有bert
    if model_config is None:
        config_override = {
            "vocabulary.directory": vocab_path,      # 改写词表的地址
            "dataset_reader.vocab_path": vocab_path,
            "model.text_field_embedder.token_embedders.pretrained_file": None  # 改写预训练词向量的地址
        }
    else:
        config_override = {
            "vocabulary.directory": vocab_path,  # 改写词表的地址
            "dataset_reader.vocab_path": vocab_path,
            "dataset_reader.model_name": model_path,
            "model.model_name": os.path.join(model_path, model_config)
        }
    archive = load_archive(os.path.join(model_path, "model.tar.gz"),
                           cuda_device=device,
                           overrides=json.dumps(config_override))
    predictor = Predictor.from_archive(archive, predictor_name)
    return predictor
