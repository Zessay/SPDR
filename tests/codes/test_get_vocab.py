# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-08
"""
测试获取allennlp词表文件
"""
from pathlib import Path
from allennlp.data import Vocabulary

basename = "/home/zs261988/"
save_path = "data/vocab/bert_vocabulary"
vocab_file = "models/ptms/albert_void_tiny/vocab.txt"

# vocab = Vocabulary(padding_token="[PAD]", oov_token="[UNK]")
# # #
# # # 加载bert词表
# vocab.set_from_file(Path(basename) / vocab_file, oov_token="[UNK]")
# # #
# vocab.save_to_files(Path(basename) / save_path)
#
# 加载之前保存到词表
vocab = Vocabulary.from_files(Path(basename) / save_path,
                              padding_token="[PAD]",
                              oov_token="[UNK]")

print("oov_token: ", vocab._oov_token, vocab.get_token_index(vocab._oov_token))
print("padding_token: ", vocab._padding_token, vocab.get_token_index(vocab._padding_token))