load("//python/tflite_micro/signal:tflm_signal.bzl", "tflm_signal_kernel_library")

package(
    default_visibility = ["//python/tflite_micro/signal:__subpackages__"],
    licenses = ["notice"],
)

tflm_signal_kernel_library(
    name = "delay_op",
    srcs = ["delay_op.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "energy_op",
    srcs = ["energy_op.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "fft_ops",
    srcs = ["fft_ops.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "filter_bank_ops",
    srcs = ["filter_bank_ops.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "framer_op",
    srcs = ["framer_op.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "overlap_add_op",
    srcs = ["overlap_add_op.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "pcan_op",
    srcs = ["pcan_op.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "stacker_op",
    srcs = ["stacker_op.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "window_op",
    srcs = ["window_op.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)
