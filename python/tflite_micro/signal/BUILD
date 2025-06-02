load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//python/tflite_micro/signal:tflm_signal.bzl", "py_tflm_signal_library")
load("//tensorflow:extra_rules.bzl", "tflm_signal_friends")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = [":__subpackages__"],
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
        ":delay_op_cc",
        ":energy_op_cc",
        ":fft_ops_cc",
        ":filter_bank_ops_cc",
        ":framer_op_cc",
        ":overlap_add_op_cc",
        ":pcan_op_cc",
        ":stacker_op_cc",
        ":window_op_cc",
    ],
)

py_library(
    name = "ops",
    srcs = [
        "__init__.py",
        "ops/__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":delay_op",
        ":energy_op",
        ":fft_ops",
        ":filter_bank_ops",
        ":framer_op",
        ":overlap_add_op",
        ":pcan_op",
        ":stacker_op",
        ":window_op",
    ],
)

py_tflm_signal_library(
    name = "delay_op",
    srcs = ["ops/delay_op.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:delay_op"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:delay_kernel",
    ],
    deps = [
        "//python/tflite_micro/signal/utils:util",
    ],
)

py_test(
    name = "delay_op_test",
    size = "small",
    srcs = ["ops/delay_op_test.py"],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":delay_op",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_tflm_signal_library(
    name = "energy_op",
    srcs = ["ops/energy_op.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:energy_op"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:energy_kernel",
    ],
    deps = [
        "//python/tflite_micro/signal/utils:util",
    ],
)

py_test(
    name = "energy_op_test",
    size = "small",
    srcs = ["ops/energy_op_test.py"],
    data = [
        "//python/tflite_micro/signal/ops/testdata:energy_test1.txt",
    ],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":energy_op",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_tflm_signal_library(
    name = "fft_ops",
    srcs = ["ops/fft_ops.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:fft_ops"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:fft_kernel",
    ],
    deps = [
        "//python/tflite_micro/signal/utils:util",
    ],
)

py_test(
    name = "fft_ops_test",
    srcs = ["ops/fft_ops_test.py"],
    data = [
        "//python/tflite_micro/signal/ops/testdata:fft_auto_scale_test1.txt",
        "//python/tflite_micro/signal/ops/testdata:rfft_test1.txt",
    ],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":fft_ops",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_tflm_signal_library(
    name = "filter_bank_ops",
    srcs = ["ops/filter_bank_ops.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:filter_bank_ops"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:filter_bank_kernels",
    ],
    deps = [
        "//python/tflite_micro/signal/utils:freq_to_mel",
        "//python/tflite_micro/signal/utils:util",
    ],
)

py_test(
    name = "filter_bank_ops_test",
    size = "small",
    srcs = ["ops/filter_bank_ops_test.py"],
    data = [
        "//python/tflite_micro/signal/ops/testdata:filter_bank_accumulation_16k.txt",
        "//python/tflite_micro/signal/ops/testdata:filter_bank_accumulation_44k.txt",
        "//python/tflite_micro/signal/ops/testdata:filter_bank_accumulation_8k.txt",
        "//python/tflite_micro/signal/ops/testdata:filter_bank_spectral_subtraction_test1.txt",
        "//python/tflite_micro/signal/ops/testdata:filter_bank_square_root_test1.txt",
        "//python/tflite_micro/signal/ops/testdata:filter_bank_test1.txt",
    ],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":filter_bank_ops",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_tflm_signal_library(
    name = "framer_op",
    srcs = ["ops/framer_op.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:framer_op"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:framer_kernel",
    ],
    deps = [
        "//python/tflite_micro/signal/utils:util",
    ],
)

py_test(
    name = "framer_op_test",
    size = "small",
    srcs = ["ops/framer_op_test.py"],
    data = [
        "//python/tflite_micro/signal/ops/testdata:framer_test1.txt",
    ],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":framer_op",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_tflm_signal_library(
    name = "overlap_add_op",
    srcs = ["ops/overlap_add_op.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:overlap_add_op"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:overlap_add_kernel",
    ],
    deps = [
        "//python/tflite_micro/signal/utils:util",
    ],
)

py_test(
    name = "overlap_add_op_test",
    size = "small",
    srcs = ["ops/overlap_add_op_test.py"],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":overlap_add_op",
        "@absl_py//absl/testing:parameterized",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_tflm_signal_library(
    name = "pcan_op",
    srcs = ["ops/pcan_op.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:pcan_op"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:pcan_kernel",
    ],
    deps = [
        "//python/tflite_micro/signal/utils:util",
        "//python/tflite_micro/signal/utils:wide_dynamic_func_lut",
    ],
)

py_test(
    name = "pcan_op_test",
    srcs = ["ops/pcan_op_test.py"],
    data = [
        "//python/tflite_micro/signal/ops/testdata:pcan_op_test1.txt",
    ],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":pcan_op",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_tflm_signal_library(
    name = "stacker_op",
    srcs = ["ops/stacker_op.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:stacker_op"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:stacker_kernel",
    ],
    deps = [
        "//python/tflite_micro/signal/utils:util",
    ],
)

py_test(
    name = "stacker_op_test",
    size = "small",
    srcs = ["ops/stacker_op_test.py"],
    data = [
        "//python/tflite_micro/signal/ops/testdata:stacker_test1.txt",
    ],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":stacker_op",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_tflm_signal_library(
    name = "window_op",
    srcs = ["ops/window_op.py"],
    cc_op_defs = ["//signal/tensorflow_core/ops:window_op"],
    cc_op_kernels = [
        "//signal/tensorflow_core/kernels:window_kernel",
    ],
    deps = [
        "//python/tflite_micro/signal/utils:util",
    ],
)

py_test(
    name = "window_op_test",
    srcs = ["ops/window_op_test.py"],
    data = [
        "//python/tflite_micro/signal/ops/testdata:window_test1.txt",
    ],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":window_op",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)
