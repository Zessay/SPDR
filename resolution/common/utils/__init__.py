# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-27
from resolution.common.utils.utils_span import get_best_span, replace_masked_values_with_big_negative_number
from resolution.common.utils.utils_allennlp import seed_everything, init_logger
from resolution.common.utils.utils_rewrite import Hypothesis, run_beam_search, convert_indices_to_string