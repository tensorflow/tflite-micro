package(
    default_visibility = ["//tensorflow/lite/micro/kernels:__pkg__"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
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

cc_library(
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
