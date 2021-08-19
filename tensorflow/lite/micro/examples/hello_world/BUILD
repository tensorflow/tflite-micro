# Description:
#   TensorFlow Lite for Microcontrollers "hello world" example.
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "micro_copts",
)

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

generate_cc_arrays(
    name = "generated_hello_world_model_cc",
    src = "hello_world.tflite",
    out = "hello_world_model_data.cc",
)

generate_cc_arrays(
    name = "generated_hello_world_model_hdr",
    src = "hello_world.tflite",
    out = "hello_world_model_data.h",
)

cc_library(
    name = "model",
    srcs = [
        ":generated_hello_world_model_cc",
    ],
    hdrs = [
        ":generated_hello_world_model_hdr",
    ],
    copts = micro_copts(),
)

cc_test(
    name = "hello_world_test",
    srcs = [
        "hello_world_test.cc",
    ],
    deps = [
        ":model",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
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
    copts = micro_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
    ],
)

cc_library(
    name = "constants",
    srcs = [
        "constants.cc",
    ],
    hdrs = [
        "constants.h",
    ],
    copts = micro_copts(),
)

cc_binary(
    name = "hello_world",
    srcs = [
        "main.cc",
        "main_functions.cc",
        "main_functions.h",
    ],
    copts = [
        "-Werror",
        "-Wsign-compare",
    ],
    deps = [
        ":constants",
        ":model",
        ":output_handler",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:system_setup",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

sh_test(
    name = "hello_world_binary_test",
    srcs = ["hello_world_binary_test.sh"],
    data = [":hello_world"],
)
