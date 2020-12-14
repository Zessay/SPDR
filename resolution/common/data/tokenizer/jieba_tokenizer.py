# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-12
import jieba
from typing import List, Optional
from overrides import overrides

from allennlp.data.tokenizers import Tokenizer
from resolution.common.data.tokenizer import TokenAdd


@Tokenizer.register("jieba")
class JiebaTokenizer(Tokenizer):
    def __init__(self,
                 do_lowercase: bool = True,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 never_split: Optional[List[str]] = None):
        self._do_lowercase = do_lowercase
        self._never_split = never_split or []
        # jieba初始化
        jieba.initialize()
        # 有些词是不希望分割的
        # 包含标签符号的词是无效的，比如[CLS]
        for word in self._never_split:
            jieba.suggest_freq(word, tune=True)
        self._start_tokens = start_tokens or []
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> List[TokenAdd]:
        word_list = jieba.lcut(text)
        # 将词语小写
        if self._do_lowercase:
            for i, word in enumerate(word_list):
                if word not in self._never_split:
                    word_list[i] = word.lower()
        if self._start_tokens:
            word_list = self._start_tokens + word_list
        if self._end_tokens:
            word_list += self._end_tokens
        word_list = [TokenAdd(token) for token in word_list]

        return word_list


