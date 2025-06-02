load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@flatbuffers//:runtime_py",
        requirement("tensorflow"),
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        "@flatbuffers//:runtime_py",
        requirement("tensorflow"),
        "//tensorflow/lite/python:schema_py",
    ],
)

py_binary(
    name = "strip_strings",
    srcs = ["strip_strings.py"],
    deps = [
        ":flatbuffer_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
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

py_test(
    name = "flatbuffer_utils_test",
    srcs = ["flatbuffer_utils_test.py"],
    deps = [
        ":flatbuffer_utils",
        ":test_utils",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "visualize_test",
    srcs = ["visualize_test.py"],
    deps = [
        ":test_utils",
        ":visualize",
        requirement("tensorflow"),
    ],
)
