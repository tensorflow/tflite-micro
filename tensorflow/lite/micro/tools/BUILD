load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("//tensorflow:extra_rules.bzl", "tflm_application_friends")

package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

package_group(
    name = "tflm_tools",
    packages = ["//tensorflow/lite/micro/tools/..."],
)

package_group(
    name = "application_friends",
    packages = tflm_application_friends(),
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
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    deps = [
        ":requantize_flatbuffer",
        "//python/tflite_micro:runtime",
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

py_library(
    name = "model_transforms_utils",
    srcs = ["model_transforms_utils.py"],
    data = [
        ":tflite_flatbuffer_align_wrapper.so",
    ],
    srcs_version = "PY3",
    visibility = [
        ":application_friends",
        ":tflm_tools",
    ],
    deps = [
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
    ],
)

py_library(
    name = "tflm_model_transforms_lib",
    srcs = ["tflm_model_transforms_lib.py"],
    data = [
        ":tflite_flatbuffer_align",
    ],
    srcs_version = "PY3",
    visibility = [
        ":application_friends",
        ":tflm_tools",
        "//:__subpackages__",
    ],
    deps = [
        ":model_transforms_utils",
        "//tensorflow/lite/micro/python/interpreter/src:runtime",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "@absl_py//absl/logging",
        requirement("numpy"),
        requirement("tensorflow-cpu"),
    ],
)

py_binary(
    name = "tflm_model_transforms",
    srcs = ["tflm_model_transforms.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tflm_model_transforms_lib",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "tflm_model_transforms_test",
    srcs = ["tflm_model_transforms_test.py"],
    data = [
        "//tensorflow/lite/micro/models",
    ],
    main = "tflm_model_transforms_test.py",
    python_version = "PY3",
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":tflm_model_transforms_lib",
        "//tensorflow/lite/micro/examples/recipes:resource_variables_lib",
        "@absl_py//absl/testing:parameterized",
        requirement("tensorflow-cpu"),
    ],
)
