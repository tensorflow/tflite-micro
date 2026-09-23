load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_rules.bzl", "tflm_py_binary")

package(licenses = ["notice"])

tflm_py_binary(
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
        requirement("mako"),
        "//tensorflow/lite/micro/tools:generate_test_for_model",
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
        "//tensorflow/lite/tools:flatbuffer_utils",
    ],
)
