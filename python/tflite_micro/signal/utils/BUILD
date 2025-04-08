# Signal python utilities.
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = [
        "//python/tflite_micro/signal:__subpackages__",
    ],
    licenses = ["notice"],
)

py_test(
    name = "freq_to_mel_test",
    srcs = ["freq_to_mel_test.py"],
    data = [
        ":freq_to_mel_wrapper.so",
    ],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
)

py_library(
    name = "freq_to_mel",
    data = [
        ":freq_to_mel_wrapper.so",
    ],
)

pybind_extension(
    name = "freq_to_mel_wrapper",  # :freq_to_mel_wrapper.so
    srcs = [
        "freq_to_mel_wrapper.cc",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("tensorflow"),
        "//python/tflite_micro:runtime",
    ],
)

pybind_extension(
    name = "wide_dynamic_func_lut_wrapper",  # :wide_dynamic_func_lut_wrapper.so
    srcs = [
        "wide_dynamic_func_lut_wrapper.cc",
    ],
)

py_library(
    name = "wide_dynamic_func_lut",
    data = [
        ":wide_dynamic_func_lut_wrapper.so",
    ],
)

py_test(
    name = "wide_dynamic_func_lut_test",
    srcs = ["wide_dynamic_func_lut_test.py"],
    data = [
        ":wide_dynamic_func_lut_wrapper.so",
    ],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    visibility = ["//visibility:public"],
)
