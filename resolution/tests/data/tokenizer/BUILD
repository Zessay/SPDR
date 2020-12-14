package(default_visibility = ["//visibility:public"])

load("@allennlp_100//:requirements.bzl", "all_requirements")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_test(
    name = "test_chinese_character_tokenizer",
    size = "medium",
    srcs = [
        "test_chinese_character_tokenizer.py",
    ],
    deps = all_requirements + [
        "//adabrain/summarization/resolution/common",
    ],
)
py_test(
    name = "test_pretrained_chinese_bert_tokenizer",
    size = "medium",
    srcs = [
        "test_pretrained_chinese_bert_tokenizer.py",
    ],
    data = [
        "//adabrain/tests/data:testdata"
    ],
    deps = all_requirements + [
        "//adabrain/summarization/resolution/common",
    ],
)
