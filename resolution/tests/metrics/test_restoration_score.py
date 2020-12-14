# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-25
from unittest import TestCase, main
from resolution.common.metrics import RestorationScore


class TestRestorationScore(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.restoration_score = RestorationScore()

    def test_restoration_score(self):
        # exact match
        predictions = list("腊八粥喝了吗")
        gold_targets = list("腊八粥喝了吗")
        restore_tokens = list("腊八粥")

        self.restoration_score(predictions, gold_targets,
                               restore_tokens=restore_tokens)
        metrics = self.restoration_score.get_metric()
        self.assertAlmostEqual(metrics['f3'], 1, 1)

        # test reset
        self.restoration_score.reset()
        metrics = self.restoration_score.get_metric()
        assert metrics['f1'] == 0
        assert metrics['f2'] == 0
        assert metrics['f3'] == 0

        # not exact match
        predictions = list("粥喝了吗")
        gold_targets = list("腊八粥喝了吗")
        restore_tokens = list("腊八粥")
        self.restoration_score(predictions, gold_targets,
                               restore_tokens=restore_tokens)

        metrics = self.restoration_score.get_metric()
        self.assertAlmostEqual(metrics['f1'], 0.500, 3)

    def test_compute_restore_tokens(self):
        restoration_score = RestorationScore(compute_restore_tokens=True)

        predictions = list("粥喝了吗")
        gold_targets = list("腊八粥喝了吗")
        queries = list("喝了吗")
        restoration_score(predictions, gold_targets, queries)
        metrics = restoration_score.get_metric()
        self.assertAlmostEqual(metrics['f1'], 0.500, 3)


if __name__ == '__main__':
    main()