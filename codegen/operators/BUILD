load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "operator",
    srcs = [
        "operator.py",
    ],
    deps = [
        "//codegen:utils",
        "//tensorflow/lite/python:schema_py",
    ],
)
