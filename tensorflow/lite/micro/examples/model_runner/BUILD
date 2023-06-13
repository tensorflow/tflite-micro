# Description:
#   TensorFlow Lite for Microcontrollers "gesture recognition" example.

load(
    "//tensorflow/lite/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = [
        "model.h",
    ],
)

tflite_micro_cc_test(
    name = "magic_wand_test",
    srcs = [
        "magic_wand_test.cc",
    ],
    deps = [
        ":magic_wand_model_data",
        ":sample_feature_data",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

tflite_micro_cc_test(
    name = "accelerometer_handler_test",
    srcs = [
        "accelerometer_handler_test.cc",
    ],
    deps = [
        ":accelerometer_handler",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)


tflite_micro_cc_test(
    name = "gesture_predictor_test",
    srcs = [
        "gesture_predictor_test.cc",
    ],
    deps = [
        ":constants",
        ":gesture_predictor",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "output_handler",
    srcs = [
        "output_handler.cc",
    ],
    hdrs = [
        "output_handler.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
    ],
)

tflite_micro_cc_test(
    name = "output_handler_test",
    srcs = [
        "output_handler_test.cc",
    ],
    deps = [
        ":output_handler",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_binary(
    name = "model_runner",
    srcs = [
        "main.cc",
        "main_functions.cc",
        "main_functions.h",
        "micro_api.cc",
        "micro_api.h"
    ],
    copts = [
        "-Werror",
        "-Wdouble-promotion",
        "-Wsign-compare",
    ],
    deps = [
        ":model",
        ":output_handler",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
