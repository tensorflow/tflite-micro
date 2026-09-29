load("@rules_python//python:defs.bzl", "py_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_rules.bzl", "tflm_py_library", "tflm_py_test")

tflm_py_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("flatbuffers"),
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
    ],
)

tflm_py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("flatbuffers"),
        "//tensorflow/lite/python:schema_py",
    ],
)

tflm_py_test(
    name = "flatbuffer_utils_test",
    srcs = ["flatbuffer_utils_test.py"],
    deps = [
        ":flatbuffer_utils",
        ":test_utils",
        "//tensorflow/lite/python:schema_py",
    ],
)

py_binary(
    name = "strip_strings",
    srcs = ["strip_strings.py"],
    deps = [
        ":flatbuffer_utils",
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

tflm_py_test(
    name = "visualize_test",
    srcs = ["visualize_test.py"],
    deps = [
        ":test_utils",
        ":visualize",
    ],
)
