load("@rules_python//python:defs.bzl", "py_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(licenses = ["notice"])

py_binary(
    name = "generate_per_layer_tests",
    srcs = [
        "generate_per_layer_tests.py",
    ],
    data = [
        "templates/BUILD.mako",
        "templates/integration_tests_cc.mako",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        requirement("mako"),
        requirement("tensorflow"),
        "//tensorflow/lite/micro/tools:generate_test_for_model",
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
        "//tensorflow/lite/tools:flatbuffer_utils",
    ],
)
