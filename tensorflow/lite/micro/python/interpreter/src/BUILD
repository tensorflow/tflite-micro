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
        "//tensorflow/lite/micro/integration_tests:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

pybind_extension(
    name = "interpreter_wrapper_pybind",
    # target = interpreter_wrapper_pybind.so because pybind_extension()
    # appends suffix.
    srcs = [
        "interpreter_wrapper.cc",
        "interpreter_wrapper.h",
        "interpreter_wrapper_pybind.cc",
        "numpy_utils.cc",
        "numpy_utils.h",
        "pybind11_lib.h",
        "python_utils.cc",
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
