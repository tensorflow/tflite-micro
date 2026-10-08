load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "c_api_types",
    hdrs = ["c_api_types.h"],
    deps = [
        "//tensorflow/lite/micro:micro_common",
    ],
)

tflm_cc_library(
    name = "common",
    hdrs = [
        "builtin_op_data.h",
        "common.h",
    ],
    deps = [
        ":c_api_types",
        "//tensorflow/lite/micro:builtin_op_data",
        "//tensorflow/lite/micro:micro_common",
    ],
)

tflm_cc_library(
    name = "builtin_op_data",
    hdrs = ["builtin_op_data.h"],
    deps = [
        ":c_api_types",
        ":common",
        "//tensorflow/lite/micro:builtin_op_data",
    ],
)
