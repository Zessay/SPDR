# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-28
from unittest import TestCase, main
from resolution.common.metrics import RewriteEM


class TestRewriteEM(TestCase):
    def setUp(self):
        super().setUp()
        self.rewrite_em = RewriteEM()

    def test_rewrite_em(self):
        # no-rewrite exact match
        predictions = "腊八粥喝了吗"
        gold_targets = "腊八粥喝了吗"
        queries = "腊八粥喝了吗"

        self.rewrite_em(predictions, gold_targets, queries)

        metrics = self.rewrite_em.get_metric()
        assert metrics['em'] == 1
        assert metrics['nr_em'] == 1

        # test reset
        self.rewrite_em.reset()
        metrics = self.rewrite_em.get_metric()
        assert metrics['em'] == 0
        assert metrics['semr'] == 0

        # rewrite semr
        predictions = "腊八粥喝了吗"
        gold_targets = "粥喝了吗"
        queries = "喝了吗"
        self.rewrite_em(predictions, gold_targets, queries)

        metrics = self.rewrite_em.get_metric()
        assert metrics['em'] == 0
        self.assertAlmostEqual(metrics['semr'], 0.666666, 4)


if __name__ == '__main__':
    main()
