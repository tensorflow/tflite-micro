load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/micro:build_def.bzl", "micro_copts")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "api",
    srcs = [
        "flatbuffer_conversions.cc",
        "tensor_utils.cc",
    ],
    hdrs = [
        "error_reporter.h",
        "flatbuffer_conversions.h",
        "op_resolver.h",
        "tensor_utils.h",
    ],
    copts = tflite_copts() + micro_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":error_reporter",
        ":op_resolver",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

# We define separate targets for "op_resolver" and "error_reporter",
# even though those headers are also exported by the "api" target,
# so that targets which only want to depend on these small abstract base
# class modules can express more fine-grained dependencies without
# pulling in tensor_utils and flatbuffer_conversions.

cc_library(
    name = "op_resolver",
    srcs = ["op_resolver.cc"],
    hdrs = ["op_resolver.h"],
    copts = tflite_copts() + micro_copts(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":error_reporter",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "error_reporter",
    srcs = ["error_reporter.cc"],
    hdrs = ["error_reporter.h"],
    copts = tflite_copts() + micro_copts(),
    visibility = [
        "//visibility:public",
    ],
    deps = [],
)
