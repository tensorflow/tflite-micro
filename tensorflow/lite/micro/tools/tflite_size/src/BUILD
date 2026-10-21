load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load(
    "//python:py_rules.bzl",
    "tflm_py_binary",
    "tflm_py_library",
)
load("//tensorflow/lite/micro:build_def.bzl", "tflm_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "flatbuffer_size_wrapper_lib",
    srcs = [
        "flatbuffer_size.cc",
        "flatbuffer_size_wrapper.cc",
    ],
    hdrs = [
        "flatbuffer_size.h",
        "flatbuffer_size_wrapper.h",
    ],
    deps = [
        "//tensorflow/lite/micro/schema:schema_fbs_with_reflection",
        "@flatbuffers",
    ],
    alwayslink = True,
)

cc_binary(
    name = "libflatbuffer_size.so",
    linkshared = True,
    deps = [
        ":flatbuffer_size_wrapper_lib",
    ],
)

tflm_py_library(
    name = "flatbuffer_size_lib",
    srcs = [
        "flatbuffer_size_graph.py",
        "flatbuffer_size_graph_html_converter.py",
    ],
    data = [
        ":libflatbuffer_size.so",
    ],
    deps = [
        "//python/tflite_micro:runtime",
    ],
)

tflm_py_library(
    name = "flatbuffer_size",
    srcs = ["flatbuffer_size.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":flatbuffer_size_lib",
    ],
)

tflm_py_binary(
    name = "tflite_size",
    srcs = ["flatbuffer_size.py"],
    main = "flatbuffer_size.py",
    deps = [
        ":flatbuffer_size",
    ],
)
