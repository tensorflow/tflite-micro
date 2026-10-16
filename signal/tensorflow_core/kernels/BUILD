load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//python/tflite_micro/signal:tflm_signal.bzl", "tflm_signal_kernel_library")

package(
    default_visibility = ["//python/tflite_micro/signal:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "kernel_utils",
    srcs = ["kernel_utils.cc"],
    hdrs = ["kernel_utils.h"],
    deps = [
        "//third_party/tf_c_headers:kernels",
        "//third_party/tf_c_headers:tf_datatype",
        "//third_party/tf_c_headers:tf_status",
        "//third_party/tf_c_headers:tf_tensor",
    ],
)

tflm_signal_kernel_library(
    name = "delay_kernel",
    srcs = ["delay_kernel.cc"],
    deps = [
        ":kernel_utils",
        "//signal/src:circular_buffer",
        "//third_party/tf_c_headers:kernels",
    ],
)

tflm_signal_kernel_library(
    name = "energy_kernel",
    srcs = ["energy_kernel.cc"],
    deps = [
        ":kernel_utils",
        "//signal/src:complex",
        "//signal/src:energy",
        "//third_party/tf_c_headers:kernels",
    ],
)

tflm_signal_kernel_library(
    name = "fft_kernel",
    srcs = ["fft_kernels.cc"],
    deps = [
        ":kernel_utils",
        "//signal/src:fft_auto_scale",
        "//signal/src:irfft",
        "//signal/src:rfft",
        "//third_party/tf_c_headers:kernels",
        "//third_party/tf_c_headers:tf_datatype",
    ],
)

tflm_signal_kernel_library(
    name = "filter_bank_kernels",
    srcs = ["filter_bank_kernels.cc"],
    deps = [
        ":kernel_utils",
        "//signal/src:filter_bank",
        "//signal/src:filter_bank_log",
        "//signal/src:filter_bank_spectral_subtraction",
        "//signal/src:filter_bank_square_root",
        "//third_party/tf_c_headers:kernels",
    ],
)

tflm_signal_kernel_library(
    name = "framer_kernel",
    srcs = ["framer_kernel.cc"],
    deps = [
        ":kernel_utils",
        "//signal/src:circular_buffer",
        "//third_party/tf_c_headers:kernels",
    ],
)

tflm_signal_kernel_library(
    name = "overlap_add_kernel",
    srcs = ["overlap_add_kernel.cc"],
    deps = [
        ":kernel_utils",
        "//signal/src:overlap_add",
        "//third_party/tf_c_headers:kernels",
        "//third_party/tf_c_headers:tf_datatype",
    ],
)

tflm_signal_kernel_library(
    name = "pcan_kernel",
    srcs = ["pcan_kernel.cc"],
    deps = [
        ":kernel_utils",
        "//signal/src:pcan_argc_fixed",
        "//third_party/tf_c_headers:kernels",
    ],
)

tflm_signal_kernel_library(
    name = "stacker_kernel",
    srcs = ["stacker_kernel.cc"],
    deps = [
        ":kernel_utils",
        "//signal/src:circular_buffer",
        "//third_party/tf_c_headers:kernels",
    ],
)

tflm_signal_kernel_library(
    name = "window_kernel",
    srcs = ["window_kernel.cc"],
    deps = [
        ":kernel_utils",
        "//signal/src:window",
        "//third_party/tf_c_headers:kernels",
    ],
)
