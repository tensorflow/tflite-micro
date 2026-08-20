load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "compatibility",
    hdrs = ["compatibility.h"],
    deps = ["//tensorflow/lite/kernels:op_macros"],
)
