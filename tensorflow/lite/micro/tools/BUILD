package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "generate_cc_arrays_lib",
    srcs = ["generate_cc_arrays.py"],
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
    ],
)

cc_binary(
    name = "tflite_flatbuffer_align",
    srcs = [
        "tflite_flatbuffer_align.cc",
    ],
    deps = [
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)
