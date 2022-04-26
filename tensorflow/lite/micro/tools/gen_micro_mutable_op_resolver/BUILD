load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

py_binary(
    name = "generate_micro_mutable_op_resolver_from_model",
    srcs = [
        "generate_micro_mutable_op_resolver_from_model.py",
    ],
    data = [
        "templates/micro_mutable_op_resolver.h.mako",
    ],
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
        "//tensorflow/lite/tools:visualize",
    ],
)

py_binary(
    name = "generate_micro_mutable_op_resolver_from_model_test",
    srcs = [
        "generate_micro_mutable_op_resolver_from_model_test.py",
    ],
    data = [
        "templates/BUILD.mako",
        "templates/micro_mutable_op_resolver.h.mako",
        "templates/micro_mutable_op_resolver_test.cc.mako",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        requirement("tensorflow-cpu"),
        requirement("mako"),
        "//tensorflow/lite/micro/tools:generate_test_for_model",
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
        "//tensorflow/lite/tools:flatbuffer_utils",
    ],
)
