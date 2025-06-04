load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
)

package(
    default_visibility = ["//tensorflow/lite/micro/kernels:__pkg__"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

####################################
# C++ libraries
####################################

tflm_cc_library(
    name = "conv_test_data",
    srcs = ["conv_test_data.cc"],
    hdrs = ["conv_test_data.h"],
    deps = ["//tensorflow/lite/c:common"],
)

tflm_cc_library(
    name = "lstm_test_data",
    srcs = ["lstm_test_data.cc"],
    hdrs = [
        "lstm_test_data.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:lstm_shared",
    ],
)

####################################
# Python
####################################
py_binary(
    name = "lstm_test_data_generator",
    srcs = [
        "lstm_test_data_generator.py",
        "lstm_test_data_utils.py",
    ],
    deps = [
        "@absl_py//absl:app",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "lstm_test_data_generator_test",
    srcs = ["lstm_test_data_generator_test.py"],
    main = "lstm_test_data_generator_test.py",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    deps = [
        ":lstm_test_data_generator",
    ],
)
