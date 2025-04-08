# Description:
#   TensorFlow Lite for Microcontrollers "hello world" example.
load("@rules_python//python:defs.bzl", "py_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_cc_test",
)

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "model",
    srcs = [
        "//tensorflow/lite/micro/examples/hello_world/models:generated_hello_world_float_model_cc",
        "//tensorflow/lite/micro/examples/hello_world/models:generated_hello_world_int8_model_cc",
    ],
    hdrs = [
        "//tensorflow/lite/micro/examples/hello_world/models:generated_hello_world_float_model_hdr",
        "//tensorflow/lite/micro/examples/hello_world/models:generated_hello_world_int8_model_hdr",
    ],
)

tflm_cc_test(
    name = "hello_world_test",
    srcs = [
        "hello_world_test.cc",
    ],
    deps = [
        ":model",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_profiler",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    data = ["//tensorflow/lite/micro/examples/hello_world/models:hello_world_float.tflite"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        requirement("numpy"),
        requirement("tensorflow"),
        "//python/tflite_micro:runtime",
    ],
)

py_binary(
    name = "evaluate_test",
    srcs = ["evaluate_test.py"],
    data = [
        "//tensorflow/lite/micro/examples/hello_world/models:hello_world_float.tflite",
        "//tensorflow/lite/micro/examples/hello_world/models:hello_world_int8.tflite",
    ],
    deps = [
        ":evaluate",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)
