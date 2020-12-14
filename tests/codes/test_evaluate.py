# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-25
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../../"))

import torch
from pathlib import Path
from tqdm.auto import tqdm
from allennlp.data import Vocabulary, DataLoader
from allennlp.data.samplers import BucketBatchSampler

from resolution.common.data.reader import BertSpanResolutionReader, BertWordSpanResolutionReader
from resolution.common.models import BertSpanPointerResolution


bert_path = "/home/zs261988/models/ptms/bert_rbt3_pytorch/"
pretrained_file = "/home/zs261988/models/mask_resolution/bert_rbt3_bs_task_expand/"
max_turn_len = 3
max_length = 256

validation_data_path="/home/zs261988/data/rewrite/business/mask_alipay_val.txt"

# 构建词表
print("加载词表.........")
vocab = Vocabulary(padding_token="[PAD]", oov_token="[UNK]")
vocab.set_from_file(bert_path + "vocab.txt",
                    is_padded=False, oov_token="[UNK]", namespace="bert_tags")

# 构架reader和模型
print("定义模型........")
reader = BertSpanResolutionReader(model_name=bert_path,
                                  max_turn_len=max_turn_len,
                                  max_length=max_length)
model = BertSpanPointerResolution(vocab=vocab,
                                  model_name=bert_path,
                                  max_turn_len=max_turn_len,
                                  task_pretrained_file=Path(pretrained_file) / "best.th")
model = model.eval()

# 读取测试集数据
instances = reader.read(validation_data_path)
instances.vocab = vocab

datasampler = BucketBatchSampler(instances, batch_size=16)
dataloader = DataLoader(dataset=instances, batch_sampler=datasampler)

print("预测.........")
# 读取数据并前向传播
with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader)):
        output_dict = model(**batch)

print("所有指标：", model.get_metrics())