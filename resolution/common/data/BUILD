package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "data",
    srcs = glob([
        "*.py",
    ]),
    deps = [
        "//adabrain/summarization/resolution/common/data/token_indexer",
        "//adabrain/summarization/resolution/common/data/tokenizer",
        "//adabrain/summarization/resolution/common/data/preprocessor",
        "//adabrain/summarization/resolution/common/data/reader"
    ]
)