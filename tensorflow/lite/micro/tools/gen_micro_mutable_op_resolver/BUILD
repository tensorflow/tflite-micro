load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_rules.bzl", "tflm_py_binary")

package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

tflm_py_binary(
    name = "generate_micro_mutable_op_resolver_from_model",
    srcs = [
        "generate_micro_mutable_op_resolver_from_model.py",
    ],
    data = [
        "templates/micro_mutable_op_resolver.h.mako",
    ],
    deps = [
        requirement("mako"),
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/lite/tools:visualize",
    ],
)

tflm_py_binary(
    name = "generate_micro_mutable_op_resolver_from_model_test",
    srcs = [
        "generate_micro_mutable_op_resolver_from_model_test.py",
    ],
    data = [
        "templates/BUILD.mako",
        "templates/micro_mutable_op_resolver.h.mako",
        "templates/micro_mutable_op_resolver_test.cc.mako",
    ],
    deps = [
        requirement("mako"),
        "//tensorflow/lite/micro/tools:generate_test_for_model",
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
        "//tensorflow/lite/tools:flatbuffer_utils",
    ],
)
