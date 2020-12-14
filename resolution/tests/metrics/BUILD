package(default_visibility = ["//visibility:public"])

load("@allennlp_100//:requirements.bzl", "all_requirements")
load("//:tools/build_rules/pytest_suite.bzl", "pytest_suite")

pytest_suite(
    name = "metrics_tests",
    size = "medium",
    tests = glob(["test_*.py")
    deps = all_requirements + [
        "//adabrain/summarization/resolution/common",
    ],
)
