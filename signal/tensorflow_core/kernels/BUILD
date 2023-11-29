load("//python/tflite_micro/signal:tflm_signal.bzl", "tflm_signal_kernel_library")

package(
    default_visibility = ["//python/tflite_micro/signal:__subpackages__"],
    licenses = ["notice"],
)

tflm_signal_kernel_library(
    name = "delay_kernel",
    srcs = ["delay_kernel.cc"],
    deps = [
        "//signal/src:circular_buffer",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "energy_kernel",
    srcs = ["energy_kernel.cc"],
    deps = [
        "//signal/src:complex",
        "//signal/src:energy",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "fft_kernel",
    srcs = ["fft_kernels.cc"],
    deps = [
        "//signal/src:fft_auto_scale",
        "//signal/src:irfft",
        "//signal/src:rfft",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "filter_bank_kernels",
    srcs = ["filter_bank_kernels.cc"],
    deps = [
        "//signal/src:filter_bank",
        "//signal/src:filter_bank_log",
        "//signal/src:filter_bank_spectral_subtraction",
        "//signal/src:filter_bank_square_root",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "framer_kernel",
    srcs = ["framer_kernel.cc"],
    deps = [
        "//signal/src:circular_buffer",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "overlap_add_kernel",
    srcs = ["overlap_add_kernel.cc"],
    deps = [
        "//signal/src:overlap_add",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "pcan_kernel",
    srcs = ["pcan_kernel.cc"],
    deps = [
        "//signal/src:pcan_argc_fixed",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "stacker_kernel",
    srcs = ["stacker_kernel.cc"],
    deps = [
        "//signal/src:circular_buffer",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "window_kernel",
    srcs = ["window_kernel.cc"],
    deps = [
        "//signal/src:window",
        "@tensorflow_cc_deps//:cc_library",
    ],
)
