load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)
load(
    "//tensorflow:extra_rules.bzl",
    "tflm_python_op_resolver_friends",
)

package(
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "op_resolver_friends",
    packages = tflm_python_op_resolver_friends(),
)

cc_library(
    name = "python_ops_resolver",
    srcs = [
        "python_ops_resolver.cc",
    ],
    hdrs = [
        "python_ops_resolver.h",
    ],
    copts = micro_copts(),
    visibility = [
        ":op_resolver_friends",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

# Append _lib at the end to avoid naming collision with the extension below
# because internal tool appends a _pybind suffix.
pybind_library(
    name = "interpreter_wrapper_lib",
    srcs = [
        "interpreter_wrapper.cc",
        "numpy_utils.cc",
        "python_utils.cc",
    ],
    hdrs = [
        "interpreter_wrapper.h",
        "numpy_utils.h",
        "pybind11_lib.h",
        "python_utils.h",
        "shared_library.h",
    ],
    deps = [
        ":python_ops_resolver",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "@numpy_cc_deps//:cc_headers",
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
        ":interpreter_wrapper_lib",
    ],
)

py_library(
    name = "tflm_runtime",
    srcs = ["tflm_runtime.py"],
    data = [":interpreter_wrapper_pybind.so"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
        "//tensorflow/lite/tools:flatbuffer_utils",
    ],
)
