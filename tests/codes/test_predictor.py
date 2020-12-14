# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-07
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../../"))

import time
from resolution.common.data.reader.bert_word_span_resolution_reader import BertWordSpanResolutionReader
from resolution.common.models import BertSpanPointerResolution
from resolution.common.predictors.bert_span_resolution_predictor import load_model


model_path = "/home/zs261988/models/online/albert_tiny_word/bert4sr_model"
predictor_name = "bert_span_resolution"

predictor = load_model(model_path, predictor_name)

context = "你好<EOS>想租那个，充电宝<EOS>我下了个单"
query = "怎么支付"

instances = [{"context": "你好<EOS>想租那个，充电宝<EOS>我下了个单",
              "query": "怎么支付"},
             {"context": "你知道ETC吗<EOS>我想了解一下",
              "query": "如何办理"}]

start = time.time()
# result = predictor.predict(context, query)
result = predictor.predict_batch_json(instances)
print("Time Cost: {} ms".format((time.time()-start)*1000))

print(result)