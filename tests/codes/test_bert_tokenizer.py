# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-10

from transformers import BertTokenizer

model_name="/home/zs261988/models/ptms/albert_void_tiny"

tokenizer = BertTokenizer.from_pretrained(model_name)

print("[SEP]: ", tokenizer.convert_tokens_to_ids("[SEP]"))