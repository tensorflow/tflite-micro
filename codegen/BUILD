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

py_binary(
    name = "code_generator",
    srcs = [
        "code_generator.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graph",
        ":inference_generator",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
