# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-24
from typing import List, Dict, Optional
from overrides import overrides

from allennlp.training.metrics.metric import Metric


@Metric.register("restoration_score")
class RestorationScore(Metric):
    def __init__(self,
                 beta: float = 1.0,
                 mode: str = "1,2,3",
                 compute_restore_tokens: bool = False):
        """
        :param beta:
        :param mode:
        :param compute_restore_tokens: bool型，表示是否要计算restore tokens
        """
        self._beta = beta   # 表示 f_beta
        self._mode = mode
        self._keys = []
        if "1" in self._mode:
            self._keys.extend(['p1', 'r1', 'f1'])
        if "2" in self._mode:
            self._keys.extend(['p2', 'r2', 'f2'])
        if "3" in self._mode:
            self._keys.extend(['p3', 'r3', 'f3'])
        # 保存所有的分数
        self._scores = []
        self._compute_restore_tokens = compute_restore_tokens

    @overrides
    def __call__(self, predictions: List[str],
                 gold_targets: List[str],
                 queries: Optional[List[str]] = None,
                 restore_tokens: Optional[List[str]] = None):
        # predictions, gold_targets, queries的类型必须一致
        if len(predictions) <= 0 or len(gold_targets) <= 0:
            return
        # 必须存在restore的单词
        if restore_tokens is None or len(restore_tokens) <= 0:
            # 计算差集
            if self._compute_restore_tokens:
                restore_tokens = list(set(gold_targets) - set(queries))
                if len(restore_tokens) <= 0:
                    return
            else:
                return
        self._scores.append(self._process_row(predictions,
                                              gold_targets,
                                              restore_tokens))

    def _ngrams(self, tokens: List[str], ngram_size: int, restore_tokens: List[str]):
        ngrams = []
        # 如果单词数量不够
        if ngram_size > len(tokens):
            return ngrams

        for i in range(len(tokens) - ngram_size + 1):
            ngram = "".join(tokens[i:i+ngram_size])
            # 判断restore_tokens中是否有token存在于ngram
            # 如果存在则添加
            for token in restore_tokens:
                if token in ngram:
                    ngrams.append(ngram)
                    break
        return ngrams

    def _process_row(self,
                     predicted_row: List[str],
                     reference_row: List[str],
                     restore_tokens: List[str]):
        res = {}
        if '1' in self._mode:
            pred_ngrams, ref_ngrams = set(self._ngrams(predicted_row, 1, restore_tokens)), set(
                self._ngrams(reference_row, 1, restore_tokens))
            for k, v in self._cal_score(len(pred_ngrams & ref_ngrams),
                                        len(pred_ngrams),
                                        len(ref_ngrams)).items():
                res[k + '1'] = v
        if '2' in self._mode:
            pred_ngrams, ref_ngrams = set(self._ngrams(predicted_row, 2, restore_tokens)), set(
                self._ngrams(reference_row, 2, restore_tokens))
            for k, v in self._cal_score(len(pred_ngrams & ref_ngrams),
                                        len(pred_ngrams),
                                        len(ref_ngrams)).items():
                res[k + '2'] = v
        if '3' in self._mode:
            pred_ngrams, ref_ngrams = set(self._ngrams(predicted_row, 3, restore_tokens)), set(
                self._ngrams(reference_row, 3, restore_tokens))
            for k, v in self._cal_score(len(pred_ngrams & ref_ngrams),
                                        len(pred_ngrams),
                                        len(ref_ngrams)).items():
                res[k + '3'] = v
        return {k: res[k] for k in self._keys}

    def _cal_score(self, overlap_cnt: int, cand_cnt: int,
                   ref_cnt: int) -> Dict[str, float]:
        p = 0 if cand_cnt == 0 else overlap_cnt / cand_cnt
        r = 0 if ref_cnt == 0 else overlap_cnt / ref_cnt
        f = (1.0 + self._beta * self._beta) * p * r / (
            (self._beta * self._beta) * (p + r + 1e-8))
        return {'p': p, 'r': r, 'f': f}

    def _average(self, score_list: List[float]):
        return sum(score_list) / len(score_list)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if len(self._scores) == 0:
            result = {k: 0.0 for k in self._keys}
        else:
            keys = self._scores[0].keys()
            result = {}
            for key in keys:
                result[key] = self._average(
                    [item[key] for item in self._scores])
        if reset:
            self.reset()
        return result

    @overrides
    def reset(self) -> None:
        self._scores = []