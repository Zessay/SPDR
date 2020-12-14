# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-25
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../../"))

from resolution.common.predictors.bert_span_resolution_predictor import load_model

model_path = "/home/zs261988/models/online/albert_tiny_char/bert4sr_model"
predictor_name = "bert_span_resolution"

model = load_model(model_path, predictor_name)

print("模型加载结束")