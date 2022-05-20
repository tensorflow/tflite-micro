load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//tensorflow/lite/micro/tools:__subpackages__"],
    licenses = ["notice"],
)

pybind_library(
    name = "interpreter_wrapper",
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

# pybind_extension() appends ".so" to "name" so the actual target name contains
# the ".so" suffix
pybind_extension(
    name = "interpreter_wrapper_pybind",
    srcs = [
        "interpreter_wrapper_pybind.cc",
    ],
    deps = [
        ":interpreter_wrapper",
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
