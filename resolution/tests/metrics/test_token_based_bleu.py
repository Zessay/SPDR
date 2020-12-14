# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-11
from unittest import TestCase, main

from resolution.common.metrics import TokenBasedBLEU


class TestTokenBasedBLEU(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.token_based_bleu = TokenBasedBLEU()

    def test_token_based_bleu(self):
        predictions = list("腊八粥喝了吗")
        gold_targets = list("腊八粥喝了吗")

        self.token_based_bleu(predictions, gold_targets)
        metrics = self.token_based_bleu.get_metric()
        assert metrics["bleu_1"] == 1.0
        assert metrics["bleu_2"] == 1.0
        assert metrics["bleu_4"] == 1.0

        # test reset
        self.token_based_bleu.reset()
        metrics = self.token_based_bleu.get_metric()
        assert metrics["bleu_1"] == 0.0
        assert metrics["bleu_2"] == 0.0
        assert metrics["bleu_4"] == 0.0

        # rewrite semr
        predictions = list("腊八粥喝了吗")
        gold_targets = list("粥喝了吗")
        self.token_based_bleu(predictions, gold_targets)
        metrics = self.token_based_bleu.get_metric()
        self.assertAlmostEqual(metrics["bleu_1"], 0.6666, 2)
        self.assertAlmostEqual(metrics["bleu_2"], 0.6325, 2)
        self.assertAlmostEqual(metrics["bleu_4"], 0.5081, 2)

if __name__ == '__main__':
    main()


