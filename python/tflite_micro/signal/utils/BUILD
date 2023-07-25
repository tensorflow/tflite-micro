# Signal python utilities.
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
    python_version = "PY3",
    srcs_version = "PY3",
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
    deps = [
        "//python/tflite_micro:runtime",
        "//python/tflite_micro/signal:ops",
        requirement("tensorflow-cpu"),
    ],
)
