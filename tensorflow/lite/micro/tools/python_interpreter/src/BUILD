load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

pybind_library(
    name = "python_interpreter",
    srcs = [
        "interpreter_wrapper.cc",
        "numpy_utils.cc",
        "python_utils.cc",
    ],
    hdrs = [
        "interpreter_wrapper.h",
        "numpy_utils.h",
        "python_utils.h",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
    ],
)

pybind_extension(
    name = "interpreter_wrapper_pybind",
    srcs = [
        "interpreter_wrapper_pybind.cc",
    ],
    deps = [
        ":python_interpreter",
    ],
)

py_library(
    name = "tflm_runtime",
    srcs = ["tflm_runtime.py"],
    data = [
        ":interpreter_wrapper_pybind.so",
    ],
    srcs_version = "PY3",
)
