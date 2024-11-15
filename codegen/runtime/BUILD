load("//tensorflow/lite/micro:build_def.bzl", "tflm_cc_library")

package(default_visibility = ["//visibility:public"])

tflm_cc_library(
    name = "micro_codegen_context",
    srcs = ["micro_codegen_context.cc"],
    hdrs = ["micro_codegen_context.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro:micro_context",
        "//tensorflow/lite/micro:micro_graph",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:span",
    ],
)
