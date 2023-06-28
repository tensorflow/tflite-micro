load("//python/tflite_micro/signal:tflm_signal.bzl", "py_tflm_signal_library")
load("//tensorflow:extra_rules.bzl", "tflm_signal_friends")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    licenses = ["notice"],
)

package_group(
    name = "signal_friends",
    packages = tflm_signal_friends(),
)

cc_library(
    name = "ops_lib",
    visibility = [":signal_friends"],
    deps = [
        ":fft_ops_cc",
        ":window_op_cc",
    ],
)

py_library(
    name = "ops",
    srcs = [
        "__init__.py",
        "ops/__init__.py",
    ],
    srcs_version = "PY3",
    visibility = ["//python/tflite_micro/signal/utils:__subpackages__"],
    deps = [
        ":fft_ops",
        ":window_op",
    ],
)

py_tflm_signal_library(
    name = "fft_ops",
    srcs = ["ops/fft_ops.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:fft_ops"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:fft_kernel",
    ],
)

py_test(
    name = "fft_ops_test",
    srcs = ["ops/fft_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fft_ops",
        "//python/tflite_micro/signal/utils:util",
        requirement("numpy"),
        requirement("tensorflow-cpu"),
    ],
)

py_tflm_signal_library(
    name = "window_op",
    srcs = ["ops/window_op.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:window_op"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:window_kernel",
    ],
)

py_test(
    name = "window_op_test",
    srcs = ["ops/window_op_test.py"],
    data = [
        "//python/tflite_micro/signal/ops/testdata:window_test1.txt",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":window_op",
        "//python/tflite_micro/signal/utils:util",
        requirement("numpy"),
        requirement("tensorflow-cpu"),
    ],
)
