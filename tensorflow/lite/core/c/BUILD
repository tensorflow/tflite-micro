load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_copts",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = [
        "builtin_op_data.h",
        "common.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":c_api_types",
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
    ],
)

cc_library(
    name = "c_api_types",
    hdrs = ["c_api_types.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
    ],
)
