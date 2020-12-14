package(default_visibility = ["//visibility:public"])

load("@allennlp_100//:requirements.bzl", "all_requirements")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_test(
    name = "test_bert_span_pointer_resolution",
    size = "medium",
    srcs = [
        "test_bert_span_pointer_resolution.py",
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
    name = "test_bert_pointer_for_rewrite",
    size = "medium",
    srcs = [
        "test_bert_pointer_for_rewrite.py",
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
    name = "test_lstm_pointer_for_rewrite",
    size = "medium",
    srcs = [
        "test_lstm_pointer_for_rewrite.py",
    ],
    data = [
        "//adabrain/summarization/resolution/test_fixtures:testdata",
        "//adabrain/tests/data:testdata"
    ],
    deps = all_requirements + [
        "//adabrain/summarization/resolution/common",
    ],
)