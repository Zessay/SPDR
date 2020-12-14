# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-08
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../../"))
from pathlib import Path
from allennlp.data import DataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from resolution.common.data.reader.pointer_rewrite_reader import PointerRewriteReader
from resolution.common.models.lstm_pointer_for_rewrite import LSTMPointerForRewrite
from resolution.common.modules.decoders.stacked_lstm_decoder import StackedLstmDecoder


basename = "/home/zs261988/"
data_path = "data/"
vocab_path = "vocab/bert_vocabulary"
sample_file = "rewrite/sample_100.txt"
embedding_file = "models/ptms/word2vec.txt"

# 定义datareader
reader = PointerRewriteReader(vocab_path=Path(basename) / data_path / vocab_path)
# 获取Vocabulary对象
vocab = reader.vocab
train_data = reader.read(Path(basename) / data_path / sample_file)
train_data.vocab = vocab

print("[CLS]: ", vocab.get_token_index("[CLS]"))
print("[SEP]: ", vocab.get_token_index("[SEP]"))

datasampler = BucketBatchSampler(train_data, batch_size=16)

dataloader = DataLoader(dataset=train_data, batch_sampler=datasampler)

# for i, batch in enumerate(dataloader):
#     print(batch)
#     if i > 0:
#         break

# ------------------ 构建模型 --------------------

print("加载词向量文件")
token_embedder = Embedding(embedding_dim=256,
                           projection_dim=512,
                           pretrained_file=basename + embedding_file,
                           padding_index=0,
                           vocab=vocab)
text_field_embedder = BasicTextFieldEmbedder(token_embedders={"tokens": token_embedder})
decoder = StackedLstmDecoder(decoding_dim=256, target_embedding_dim=512,
                            num_layers=1)


model = LSTMPointerForRewrite(vocab=vocab,
                              embedding_size=512,
                              encoder_hidden_size=256,
                              encoder_num_layers=4,
                              decoder=decoder,
                              decoder_num_layers=4,
                              text_field_embedder=text_field_embedder)

for i, batch in enumerate(dataloader):
    output = model(**batch)
    print(output)