load("//signal:tflm_signal.bzl", "py_tflm_signal_library")
load("//tensorflow:extra_rules.bzl", "tflm_signal_friends")

package(
    licenses = ["notice"],
)

package_group(
    name = "signal_friends",
    packages = tflm_signal_friends(),
)

cc_library(
    name = "ops_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":window_op_cc",
    ],
)

py_library(
    name = "ops",
    srcs = [
        "python/__init__.py",
        "python/ops/__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":window_op",
    ],
)

py_tflm_signal_library(
    name = "window_op",
    srcs = ["python/ops/window_op.py"],
    cc_op_defs = ["tensorflow_core/ops/window_op.cc"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:window_kernel",
    ],
)
