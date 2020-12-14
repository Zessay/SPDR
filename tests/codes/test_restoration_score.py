# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-24
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../../"))

from resolution.common.metrics import RestorationScore

metric = RestorationScore()

predictions = list("粥喝了吗")
gold_targets = list("腊八粥喝了吗")
restore_tokens = list("腊八粥")

metric(predictions, gold_targets, restore_tokens=restore_tokens)

print(metric.get_metric())
