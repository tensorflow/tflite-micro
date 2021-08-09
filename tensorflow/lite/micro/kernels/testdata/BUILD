load("//tensorflow/lite/micro:build_def.bzl", "micro_copts")
load("//tensorflow:extra_rules.bzl", "tflm_kernel_friends")

package(
    features = ["-layering_check"],
    licenses = ["notice"],
    default_visibility = ["//tensorflow/lite/micro/kernels:__pkg__"],
)

####################################
# C++ libraries
####################################

cc_library(
    name = "conv_test_data",
    srcs = ["conv_test_data.cc"],
    hdrs = ["conv_test_data.h"],
    deps = ["//tensorflow/lite/c:common"],
)

