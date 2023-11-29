load("@rules_python//python:defs.bzl", "py_library")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)
load(
    "//tensorflow:extra_rules.bzl",
    "tflm_python_op_resolver_friends",
)

package(
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "op_resolver_friends",
    packages = tflm_python_op_resolver_friends(),
)

# tflm_runtime is deprecated, please use //python/tflite_micro:runtime instead.
# TODO(b/286456378): remove once all usage is changed to the runtime target.
py_library(
    name = "tflm_runtime",
    srcs = ["tflm_runtime.py"],
    visibility = ["//visibility:public"],
    deps = ["//python/tflite_micro:runtime"],
)

# runtime is deprecated, please use //python/tflite_micro:runtime instead.
# TODO(b/286456378): remove once all usage is changed to the runtime target.
py_library(
    name = "runtime",
    srcs = ["runtime.py"],
    visibility = ["//visibility:public"],
    deps = ["//python/tflite_micro:runtime"],
)

# TODO(b/286456378): remove once all internal usage is fixed.
cc_library(
    name = "python_ops_resolver",
    srcs = [],
    hdrs = [
        "python_ops_resolver.h",
    ],
    copts = micro_copts(),
    visibility = [":op_resolver_friends"],
    deps = [
        "//python/tflite_micro:python_ops_resolver",
    ],
)
