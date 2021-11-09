load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "generate_per_layer_tests",
    srcs = ["generate_per_layer_tests.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        requirement("tensorflow-cpu"),
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/tools:flatbuffer_utils",
    ],
)
