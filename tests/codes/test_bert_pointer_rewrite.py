# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-08
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../../"))
from pathlib import Path
from allennlp.data import DataLoader
from allennlp.data.samplers import BucketBatchSampler

from resolution.common.data.reader.bert_pointer_rewrite_reader import BertPointerRewriteReader


basename = "/home/zs261988/"
data_path = "data/"
model_path = "models/ptms/"
model_name = "albert_void_tiny/"
vocab_file = "vocab.txt"
sample_file = "rewrite/sample_100.txt"

reader = BertPointerRewriteReader(model_name=basename+model_path+model_name,
                                  vocab_file=basename+model_path+model_name+vocab_file)

# 读取数据
train_data = reader.read(Path(basename) / data_path / sample_file)
# 获取Vocabulary
vocab = reader.vocab
train_data.vocab = vocab

print("[PAD]: ", vocab.get_token_index("[PAD]", namespace="bert_tags"))
print("[CLS]: ", vocab.get_token_index("[CLS]", namespace="bert_tags"))
print("[SEP]: ", vocab.get_token_index("[SEP]", namespace="bert_tags"))

datasampler = BucketBatchSampler(train_data, batch_size=16)

dataloader = DataLoader(dataset=train_data, batch_sampler=datasampler)

for i, batch in enumerate(dataloader):
    print(batch)
    if i > 0:
        break