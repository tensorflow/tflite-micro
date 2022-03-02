load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/micro:build_def.bzl", "micro_copts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "op_macros",
    hdrs = [
        "op_macros.h",
    ],
    copts = tflite_copts(),
    deps = ["//tensorflow/lite/micro:debug_log"],
)

cc_library(
    name = "kernel_util",
    srcs = [
        "kernel_util.cc",
    ],
    hdrs = [
        "kernel_util.h",
    ],
    copts = tflite_copts() + micro_copts(),
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite/kernels/internal:quantization_util",
    ],
)

cc_library(
    name = "padding",
    srcs = [],
    hdrs = ["padding.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:types",
    ],
)
