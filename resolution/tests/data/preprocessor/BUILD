package(default_visibility = ["//visibility:public"])

load("@allennlp_100//:requirements.bzl", "all_requirements")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_test(
    name = "test_preprocess_rewrite_data",
    size = "medium",
    srcs = [
        "test_preprocess_rewrite_data.py",
    ],
    deps = all_requirements + [
        "//adabrain/summarization/resolution/common",
    ],
)
