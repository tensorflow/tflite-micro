load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "tflite_c_headers",
    hdrs = glob(["tensorflow/**/*.h"]),
    includes = ["."],
)

# Targets matching //third_party/tensorflow/lite/c in Google3 for Copybara sync.
cc_library(
    name = "c_api_opaque",
    hdrs = ["tensorflow/lite/c/c_api_opaque.h"],
    deps = [":tflite_c_headers"],
)

cc_library(
    name = "common",
    hdrs = ["tensorflow/lite/c/common.h"],
    deps = [":tflite_c_headers"],
)
