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
)

tflm_cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = [
        "builtin_op_data.h",
        "common.h",
    ],
    deps = [
        ":c_api_types",
    ],
)

tflm_cc_library(
    name = "builtin_op_data",
    hdrs = ["builtin_op_data.h"],
    deps = [
        ":c_api_types",
        ":common",
    ],
)
