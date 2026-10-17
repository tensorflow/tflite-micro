load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//python/tflite_micro/signal:tflm_signal.bzl", "tflm_signal_kernel_library")

package(
    default_visibility = ["//python/tflite_micro/signal:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "shape_inference_utils",
    srcs = ["shape_inference_utils.cc"],
    hdrs = ["shape_inference_utils.h"],
    deps = [
        "//third_party/tf_c_headers:ops",
        "//third_party/tf_c_headers:tf_status",
    ],
)

tflm_signal_kernel_library(
    name = "delay_op",
    srcs = ["delay_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tf_c_headers:ops",
        "//third_party/tf_c_headers:tf_status",
    ],
)

tflm_signal_kernel_library(
    name = "energy_op",
    srcs = ["energy_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tf_c_headers:ops",
        "//third_party/tf_c_headers:tf_status",
    ],
)

tflm_signal_kernel_library(
    name = "fft_ops",
    srcs = ["fft_ops.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tf_c_headers:ops",
        "//third_party/tf_c_headers:tf_status",
    ],
)

tflm_signal_kernel_library(
    name = "filter_bank_ops",
    srcs = ["filter_bank_ops.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tf_c_headers:ops",
        "//third_party/tf_c_headers:tf_status",
    ],
)

tflm_signal_kernel_library(
    name = "framer_op",
    srcs = ["framer_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tf_c_headers:ops",
        "//third_party/tf_c_headers:tf_status",
    ],
)

tflm_signal_kernel_library(
    name = "overlap_add_op",
    srcs = ["overlap_add_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tf_c_headers:ops",
        "//third_party/tf_c_headers:tf_status",
    ],
)

tflm_signal_kernel_library(
    name = "pcan_op",
    srcs = ["pcan_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tf_c_headers:ops",
        "//third_party/tf_c_headers:tf_status",
    ],
)

tflm_signal_kernel_library(
    name = "stacker_op",
    srcs = ["stacker_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tf_c_headers:ops",
        "//third_party/tf_c_headers:tf_status",
    ],
)

tflm_signal_kernel_library(
    name = "window_op",
    srcs = ["window_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tf_c_headers:ops",
        "//third_party/tf_c_headers:tf_status",
    ],
)
