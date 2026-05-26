load("@rules_python//python:defs.bzl", "py_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_rules.bzl", "tflm_py_library")

tflm_py_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("flatbuffers"),
        requirement("tensorflow"),
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
    ],
)

py_binary(
    name = "strip_strings",
    srcs = ["strip_strings.py"],
    deps = [
        ":flatbuffer_utils",
        requirement("absl_py"),
    ],
)

py_binary(
    name = "visualize",
    srcs = ["visualize.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//tensorflow/lite/python:schema_py",
        requirement("numpy"),
    ],
)
