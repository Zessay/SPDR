# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-03
from dataclasses import dataclass
from allennlp.data.tokenizers import Token


@dataclass(init=False, repr=False)
class TokenAdd(Token):
    def __init__(self,
                 text: str = None,
                 idx: int = None,
                 idx_end: int = None,
                 lemma_: str = None,
                 pos_: str = None,
                 tag_: str = None,
                 dep_: str = None,
                 ent_type_: str = None,
                 text_id: int = None,
                 type_id: int = None,
                 **kwargs):
        super().__init__(text, idx, idx_end, lemma_, pos_,
                         tag_, dep_, ent_type_, text_id, type_id)

        for k, v in kwargs.items():
            setattr(self, k, v)


def show_token(token: Token) -> str:
    attrs = dir(token)
    token_info = ""
    for attr in attrs:
        if attr.startswith("__"):
            continue
        else:
            value = getattr(token, attr)
            token_info += f"({attr.strip('_')}: {value}) "
    return token_info
