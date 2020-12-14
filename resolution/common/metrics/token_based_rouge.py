# coding=utf-8
# @Author: yinxu.pyx
# @Date: 2020-02-24
from typing import Tuple, Dict, Set, List
from overrides import overrides
from allennlp.training.metrics.metric import Metric
from allennlp.data.tokenizers import Token


@Metric.register("token_rouge")
class TokenBasedROUGE(Metric):
    """
    Recall-Oriented Understanding for Gisting Evaluation.

    ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation. 
    It includes measures to automatically determine the quality of a summary 
    by comparing it to other (ideal) summaries created by humans. 
    The measures count the number of over-lapping units such as n-gram, 
    word sequences, and word pairs between the computer-generated summary to 
    be evaluated and the ideal summaries created by humans. 
    This paper introduces four different ROUGE measures: ROUGE-N, ROUGE-L, 
    ROUGE-W, and ROUGE-S included in the ROUGE summarization evaluation package 
    and their evaluations. 
    Three of them have been used in the Document Understanding Conference (DUC) 2004, 
    a large-scale summarization evaluation sponsored by NIST.

    Lin, Chin-Yew. "ROUGE: A Packagefor Automatic Evaluation of Summaries." 
    ProceedingsofWorkshop on Text Summarization Branches Out, 
    Post2Conference Workshop of ACL. 2004.

    # Parameters
    exclude_tokens : `Set[str]`, optional (default = None)
        Tokens to exclude when calculating ngrams. This should usually include
        the tokens of the start, end, and pad tokens.
    beta: float, optional (default = 1.0)
        Beta parameter used for calculating F score.
    mode: str, optional (default = '1p,1r,1f,2p,2r,2f,lp,lr,lf')
        This parameter defines which metric is needed. It is a sub-list of 
        ['1p', '1r', '1f', '2p', '2r', '2f', 'lp', 'lr', 'lf'] joined by ','.
    Notes
    -----
    This implementation only considers a reference set of size 1, i.e. a single
    gold target sequence for each predicted sequence.
    """
    def __init__(self,
                 exclude_tokens: Set[str] = None,
                 beta: float = 1.0,
                 mode: str = '1p,1r,1f,2p,2r,2f,lp,lr,lf') -> None:
        self._exclude_tokens = {Token(t)
                                for t in exclude_tokens
                                } if exclude_tokens is not None else set()
        self._beta = beta
        self._scores = []
        self._mode = mode
        self._keys = [
            'rouge_{}_{}'.format(key[0], key[1]) for key in mode.split(',')
        ]

    @overrides
    def reset(self) -> None:
        self._scores = []

    def _ngrams(self, tokens: List[str],
                ngram_size: int) -> List[Tuple[int, ...]]:
        ngrams = []
        if ngram_size > len(tokens):
            return ngrams

        for i in range(len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[i:i + ngram_size])
            if any(x in self._exclude_tokens for x in ngram):  # drop PAD token
                continue
            ngrams.append(ngram)
        return ngrams

    def _lcs(self, a_tokens: List[Tuple[str, ...]],
             b_tokens: List[Tuple[str, ...]]) -> int:
        '''
            Dynamic Programming for Longest Common Subsequence
        '''
        if len(a_tokens) < len(b_tokens):
            a_tokens, b_tokens = b_tokens, a_tokens
        dp = [0] * len(b_tokens)
        for ca in a_tokens:
            upleft = 0
            for j, cb in enumerate(b_tokens):
                up = dp[j]
                if ca == cb:
                    dp[j] = upleft + 1
                else:
                    dp[j] = max(up, dp[j - 1] if j > 0 else 0)
                upleft = up
        return dp[-1]

    @overrides
    def __call__(
            self,  # type: ignore
            predictions: List[str],
            gold_targets: List[str],
    ) -> None:
        self._scores.append(self._process_row(predictions, gold_targets))

    def _process_row(self, predicted_row: List[str],
                     reference_row: List[str]) -> Dict[str, float]:
        res = {}
        if '1' in self._mode:
            pred_ngrams, ref_ngrams = set(self._ngrams(predicted_row, 1)), set(
                self._ngrams(reference_row, 1))
            for k, v in self._cal_rouge(len(pred_ngrams & ref_ngrams),
                                        len(pred_ngrams),
                                        len(ref_ngrams)).items():
                res['rouge_1_' + k] = v
        if '2' in self._mode:
            pred_ngrams, ref_ngrams = set(self._ngrams(predicted_row, 2)), set(
                self._ngrams(reference_row, 2))
            for k, v in self._cal_rouge(len(pred_ngrams & ref_ngrams),
                                        len(pred_ngrams),
                                        len(ref_ngrams)).items():
                res['rouge_2_' + k] = v
        if 'l' in self._mode:
            pred_ngrams, ref_ngrams = self._ngrams(predicted_row,
                                                   1), self._ngrams(
                                                       reference_row, 1)
            overlap_cnt = self._lcs(pred_ngrams, ref_ngrams)
            for k, v in self._cal_rouge(overlap_cnt, len(pred_ngrams),
                                        len(ref_ngrams)).items():
                res['rouge_l_' + k] = v
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

    def _cal_rouge(self, overlap_cnt: int, cand_cnt: int,
                   ref_cnt: int) -> Dict[str, float]:
        p = 0 if cand_cnt == 0 else overlap_cnt / cand_cnt
        r = 0 if ref_cnt == 0 else overlap_cnt / ref_cnt
        f = (1.0 + self._beta * self._beta) * p * r / (
            (self._beta * self._beta) * (p + r + 1e-8))
        return {'p': p, 'r': r, 'f': f}
