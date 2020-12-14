# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-24
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../../"))
from allennlp.data.tokenizers import WhitespaceTokenizer

from resolution.common.data.tokenizer import ChineseCharacterTokenizer

string = " 你 好 ， 你 叫 什 么  "

tokenizer = ChineseCharacterTokenizer()

print(tokenizer.tokenize(string))