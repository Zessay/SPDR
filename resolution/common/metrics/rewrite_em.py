# coding=utf-8
# @Author: 莫冉
# @Date: 2020-06-17

from typing import List, Dict, Union
from overrides import overrides

from allennlp.training.metrics.metric import Metric


@Metric.register("rewrite_em")
class RewriteEM(Metric):
    def __init__(self,
                 valid_keys: str = "all"):
        self._re_em = 0.0
        self._re_sem = 0.0
        self._re_semr = 0.0
        self._re_count = 0.0
        self._nr_em = 0.0
        self._nr_sem = 0.0
        self._nr_semr = 0.0
        self._nr_count = 0.0
        if valid_keys == "all":
            self._valid_keys = ["em", "sem", "semr",
                                "nr_em", "nr_semr",
                                "re_em", "re_semr"]
        else:
            self._valid_keys = valid_keys.split(",")

    @overrides
    def __call__(self, predictions: Union[List[str], str],
                 gold_targets: Union[List[str], str],
                 queries: Union[List[str], str]):
        # predictions, gold_targets, queries的类型必须一致
        if len(predictions) <= 0 or len(gold_targets) <= 0 or len(queries) <= 0:
            return
        pred_string = "".join(predictions).strip() if isinstance(
            predictions, list) else predictions
        gold_string = "".join(gold_targets).strip() if isinstance(
            gold_targets, list) else gold_targets
        query_string = "".join(queries).strip() if isinstance(
            queries, list) else queries

        # No rewrite
        if gold_string == query_string:
            if pred_string == gold_string:
                self._nr_em += 1
            # if satisfy soft-em
            if self._is_soft_em(predictions, gold_targets):
                self._nr_sem += 1
                self._nr_semr += len(gold_targets) / len(predictions)
            self._nr_count += 1
        else:
            if pred_string == gold_string:
                self._re_em += 1
            # if satisfy soft-em
            if self._is_soft_em(predictions, gold_targets):
                self._re_sem += 1
                self._re_semr += len(gold_targets) / len(predictions)
            self._re_count += 1

    def _is_soft_em(self,
                    predictions: Union[List[str], str],
                    gold_targets: Union[List[str], str]):
        g_len = len(gold_targets)
        for i, ch in enumerate(gold_targets):
            try:
                cur_index = predictions.index(ch)
                cur_index += 1
                # if current word is the last of gold
                # and so is the prediction
                if i == g_len and cur_index == len(predictions):
                    return True
                predictions = predictions[cur_index:]
            except Exception:
                return False
        return True

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        nr_em, nr_semr = 0.0, 0.0
        if self._nr_count > 0:
            nr_em = self._nr_em / self._nr_count
            nr_semr = self._nr_semr / self._nr_count

        re_em, re_semr = 0.0, 0.0
        if self._re_count > 0:
            re_em = self._re_em / self._re_count
            re_semr = self._re_semr / self._re_count

        em, sem, semr = 0.0, 0.0, 0.0
        total_count = self._nr_count + self._re_count
        if total_count > 0:
            em = (self._nr_em + self._re_em) / total_count
            sem = (self._nr_sem + self._re_sem) / total_count
            semr = (self._nr_semr + self._re_semr) / total_count
        if reset:
            self.reset()
        all_results = {"em": em, "semr": semr, "sem": sem,
                       "nr_em": nr_em, "nr_semr": nr_semr,
                       "re_em": re_em, "re_semr": re_semr}
        result = {}
        for k, v in all_results.items():
            if k in self._valid_keys:
                result[k] = v
        return result

    @overrides
    def reset(self):
        self._nr_em = 0.0
        self._nr_sem = 0.0
        self._nr_semr = 0.0
        self._nr_count = 0.0
        self._re_em = 0.0
        self._re_sem = 0.0
        self._re_semr = 0.0
        self._re_count = 0.0
