package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "array",
    srcs = ["array.cc"],
    hdrs = ["array.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "type_to_tflitetype",
    hdrs = [
        "portable_type_to_tflitetype.h",
    ],
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "kernel_api",
    hdrs = [
        "builtin_op_data.h",
        "builtin_ops.h",
        "context_util.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
    ],
)
