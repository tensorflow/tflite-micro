load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "generate_cc_arrays_lib",
    srcs = ["generate_cc_arrays.py"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
    ],
)

py_library(
    name = "generate_test_for_model",
    srcs = ["generate_test_for_model.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/lite/python:schema_py",
    ],
)

py_binary(
    name = "generate_cc_arrays",
    srcs = ["generate_cc_arrays.py"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
    ],
)

py_binary(
    name = "requantize_flatbuffer",
    srcs = [
        "requantize_flatbuffer.py",
        "requantize_flatbuffer_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "@absl_py//absl:app",
    ],
)

py_test(
    name = "requantize_flatbuffer_test",
    srcs = ["requantize_flatbuffer_test.py"],
    main = "requantize_flatbuffer_test.py",
    python_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols from interpreter_wrapper_pybind.so
        "noubsan",
    ],
    deps = [
        ":requantize_flatbuffer",
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
        requirement("numpy"),
        requirement("tensorflow-cpu"),
    ],
)

pybind_extension(
    name = "tflite_flatbuffer_align_wrapper",  # :tflite_flatbuffer_align_wrapper.so
    srcs = [
        "tflite_flatbuffer_align_wrapper.cc",
    ],
    deps = [
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

py_binary(
    name = "tflite_flatbuffer_align",
    srcs = [
        "tflite_flatbuffer_align.py",
    ],
    data = [
        ":tflite_flatbuffer_align_wrapper.so",
    ],
    deps = [
        "@absl_py//absl:app",
    ],
)
