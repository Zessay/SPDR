# coding=utf-8
# @Author: 莫冉
# @Date: 2020-06-17

from typing import Dict, List, Union
from overrides import overrides
from nltk.translate.bleu_score import sentence_bleu

from allennlp.training.metrics.metric import Metric


@Metric.register("token_bleu")
class TokenBasedBLEU(Metric):
    """
    BiLingual Evaluation Understudy.

    BLEU only needs to match human judgment when averaged over a test corpus; scores
    on individual sentence will often vary from human judgments. The key to BLEU's
    success is that all systems are treated similarly and multiple human translators
    with different styles are used, so this effect cancels out in comparisons between
    systems.

    Kishore Papineni. "BLEU: a Method for Automatic Evaluation of Machine Translation."
    ACL. 2002.

    # Parameters:
    mode: str, optional (default = '1,2,4')
        This parameter defines which metric is needed. It is a sub-list of
        ['1', '2', '4'] joined by ','.
    """

    def __init__(self,
                 mode: str = '1,2,4'):
        self._mode = mode
        self._keys = [
            'bleu_{}'.format(key) for key in mode.split(',')]
        self._scores = []

    @overrides
    def reset(self) -> None:
        self._scores = []

    @overrides
    def __call__(self,
                 predictions: List[str],
                 gold_targets: List[Union[str, List[str]]]):
        """
        :param predictions: a list of str (tokens), which is the predicted result.
        :param gold_targets: a list of str or a list comprised of a list str.
                If it is a nested list, each element in the list represents a
                ground-truth result.
        :return:
        """
        if len(gold_targets) <= 0:
            raise ValueError("No golden targets")

        if isinstance(gold_targets[0], str):
            gold_targets = [gold_targets]

        self._scores.append(self._process_row(predictions, gold_targets))

    def _process_row(self,
                     predicted_row: List[str],
                     reference_row: List[List[str]]) -> Dict[str, float]:
        res = {}
        if '1' in self._mode:
            bleu_1 = sentence_bleu(reference_row, predicted_row, weights=(1.0, 0.0, 0.0, 0.0))
            res['bleu_1'] = bleu_1

        if '2' in self._mode:
            bleu_2 = sentence_bleu(reference_row, predicted_row,
                                   weights=(0.5, 0.5, 0.0, 0.0))
            res['bleu_2'] = bleu_2

        if '4' in self._mode:
            bleu_4 = sentence_bleu(reference_row, predicted_row,
                                   weights=(0.25, 0.25, 0.25, 0.25))
            res['bleu_4'] = bleu_4

        return {k: res[k] for k in self._keys}

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

    def _average(self, score_list: List[float]) -> float:
        return sum(score_list) / len(score_list)
