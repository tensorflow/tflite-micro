load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "graph",
    srcs = [
        "graph.py",
    ],
    deps = [
        ":tensor",
        ":utils",
        "//codegen/operators:factory",
        "//codegen/operators:operator",
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/tools:visualize",
    ],
)

py_library(
    name = "inference_generator",
    srcs = [
        "inference_generator.py",
    ],
    data = [
        "templates/inference.cc.mako",
        "templates/inference.h.mako",
    ],
    deps = [
        ":graph",
        requirement("mako"),
    ],
)

py_library(
    name = "tensor",
    srcs = [
        "tensor.py",
    ],
    deps = [
        ":utils",
        "//tensorflow/lite/python:schema_py",
    ],
)

py_library(
    name = "utils",
    srcs = [
        "utils.py",
    ],
)

py_binary(
    name = "code_generator",
    srcs = [
        "code_generator.py",
    ],
    deps = [
        ":graph",
        ":inference_generator",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
