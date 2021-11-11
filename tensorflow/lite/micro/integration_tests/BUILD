load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    licenses = ["notice"],
)

py_binary(
    name = "generate_per_layer_tests",
    srcs = ["generate_per_layer_tests.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        requirement("tensorflow-cpu"),
        requirement("mako"),
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
        "//tensorflow/lite/tools:flatbuffer_utils",
    ],
)
