load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.py"],
    srcs_version = "PY3",
    deps = [
        requirement("flatbuffers"),
        requirement("tensorflow"),
        "//tensorflow/lite/python:schema_py",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        requirement("flatbuffers"),
        requirement("tensorflow"),
        "//tensorflow/lite/python:schema_py",
    ],
)

py_test(
    name = "flatbuffer_utils_test",
    srcs = ["flatbuffer_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":flatbuffer_utils",
        ":test_utils",
        requirement("tensorflow"),
    ],
)
