# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-06

from allennlp.data.vocabulary import Vocabulary

vocab_file = "../data/base_bert/vocab.txt"
save_path = "../../../vocab_path"

vocab = Vocabulary(padding_token="[PAD]", oov_token="[UNK]")

vocab.set_from_file(vocab_file, is_padded=True, oov_token="[UNK]")

vocab.save_to_files(save_path)

print(vocab.get_token_index(vocab._oov_token))