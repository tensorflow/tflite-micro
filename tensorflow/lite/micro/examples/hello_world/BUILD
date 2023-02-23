# Description:
#   TensorFlow Lite for Microcontrollers "hello world" example.
load("@tflm_pip_deps//:requirements.bzl", "requirement")
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

cc_library(
    name = "model",
    srcs = [
        "//tensorflow/lite/micro/examples/hello_world/models:generated_hello_world_float_model_cc",
    ],
    hdrs = [
        "//tensorflow/lite/micro/examples/hello_world/models:generated_hello_world_float_model_hdr",
    ],
    copts = micro_copts(),
)

cc_test(
    name = "evaluate_cc_test",
    srcs = [
        "evaluate_test.cc",
    ],
    deps = [
        ":model",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
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
        "//tensorflow/lite/micro:micro_log",
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
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
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

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    data = ["//tensorflow/lite/micro/examples/hello_world/models:hello_world_float.tflite"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
    ],
)

py_binary(
    name = "evaluate_test",
    srcs = ["evaluate_test.py"],
    data = [
        "//tensorflow/lite/micro/examples/hello_world/models:hello_world_float.tflite",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":evaluate",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
    ],
)
