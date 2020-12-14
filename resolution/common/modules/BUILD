package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "modules",
    srcs = glob([
        "*.py",
    ]),
    deps = [
        "//adabrain/summarization/resolution/common/modules/token_embedders",
        "//adabrain/summarization/resolution/common/modules/encoders",
        "//adabrain/summarization/resolution/common/modules/decoders"
    ]
)