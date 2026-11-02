load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "kernel_util",
    srcs = ["kernel_util.cc"],
    hdrs = ["kernel_util.h"],
    deps = [
        "//third_party/flatbuffers:runtime_cc",
        "//third_party/tensorflow/lite/c:c_api_opaque",
        "//third_party/tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "signal_ops",
    srcs = [
        "delay.cc",
        "energy.cc",
        "fft_auto_scale.cc",
        "filter_bank.cc",
        "filter_bank_log.cc",
        "filter_bank_spectral_subtraction.cc",
        "filter_bank_square_root.cc",
        "framer.cc",
        "irfft.cc",
        "overlap_add.cc",
        "pcan.cc",
        "rfft.cc",
        "stacker.cc",
        "window.cc",
    ],
    hdrs = ["signal_ops.h"],
    deps = [
        ":kernel_util",
        "//signal/src:circular_buffer",
        "//signal/src:complex",
        "//signal/src:energy",
        "//signal/src:fft_auto_scale",
        "//signal/src:filter_bank",
        "//signal/src:filter_bank_log",
        "//signal/src:filter_bank_spectral_subtraction",
        "//signal/src:filter_bank_square_root",
        "//signal/src:irfft",
        "//signal/src:overlap_add",
        "//signal/src:pcan_argc_fixed",
        "//signal/src:rfft",
        "//signal/src:window",
        "//third_party/flatbuffers:runtime_cc",
        "//third_party/tensorflow/lite/c:c_api_opaque",
        "//third_party/tensorflow/lite/c:common",
    ],
)
