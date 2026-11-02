load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//python:py_rules.bzl",
    "tflm_tf_shape_inference_copts",
    "tflm_tf_shape_inference_deps",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//python/tflite_micro/signal:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "shape_inference_utils",
    srcs = ["shape_inference_utils.cc"],
    hdrs = ["shape_inference_utils.h"],
    copts = tflm_tf_shape_inference_copts(),
    deps = [
        "//third_party/tensorflow/c:ops",
        "//third_party/tensorflow/c:tf_status",
    ] + tflm_tf_shape_inference_deps(),
)

cc_library(
    name = "delay_op",
    srcs = ["delay_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tensorflow/c:ops",
        "//third_party/tensorflow/c:tf_status",
    ],
    alwayslink = True,
)

cc_library(
    name = "energy_op",
    srcs = ["energy_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tensorflow/c:ops",
        "//third_party/tensorflow/c:tf_status",
    ],
    alwayslink = True,
)

cc_library(
    name = "fft_ops",
    srcs = ["fft_ops.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tensorflow/c:ops",
        "//third_party/tensorflow/c:tf_status",
    ],
    alwayslink = True,
)

cc_library(
    name = "filter_bank_ops",
    srcs = ["filter_bank_ops.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tensorflow/c:ops",
        "//third_party/tensorflow/c:tf_status",
    ],
    alwayslink = True,
)

cc_library(
    name = "framer_op",
    srcs = ["framer_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tensorflow/c:ops",
        "//third_party/tensorflow/c:tf_status",
    ],
    alwayslink = True,
)

cc_library(
    name = "overlap_add_op",
    srcs = ["overlap_add_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tensorflow/c:ops",
        "//third_party/tensorflow/c:tf_status",
    ],
    alwayslink = True,
)

cc_library(
    name = "pcan_op",
    srcs = ["pcan_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tensorflow/c:ops",
        "//third_party/tensorflow/c:tf_status",
    ],
    alwayslink = True,
)

cc_library(
    name = "stacker_op",
    srcs = ["stacker_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tensorflow/c:ops",
        "//third_party/tensorflow/c:tf_status",
    ],
    alwayslink = True,
)

cc_library(
    name = "window_op",
    srcs = ["window_op.cc"],
    deps = [
        ":shape_inference_utils",
        "//third_party/tensorflow/c:ops",
        "//third_party/tensorflow/c:tf_status",
    ],
    alwayslink = True,
)
