package(default_visibility = ["//visibility:public"])

load("@allennlp_100//:requirements.bzl", "all_requirements")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_test(
    name = "test_bert_span_resolution_predictor",
    size = "medium",
    srcs = [
        "test_bert_span_resolution_predictor.py",
    ],
    data = [
        "//adabrain/summarization/resolution/test_fixtures:testdata",
        "//adabrain/tests/data:testdata"
    ],
    deps = all_requirements + [
        "//adabrain/summarization/resolution/common",
    ],
)
py_test(
    name = "test_pointer_rewrite_predictor",
    size = "medium",
    srcs = [
        "test_pointer_rewrite_predictor.py",
    ],
    data = [
        "//adabrain/summarization/resolution/test_fixtures:testdata",
        "//adabrain/tests/data:testdata"
    ],
    deps = all_requirements + [
        "//adabrain/summarization/resolution/common",
    ],
)