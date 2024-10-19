load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_cc_test",
)
load(
    "//tensorflow:extra_rules.bzl",
    "tflm_kernel_friends",
)

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "tflite_micro",
    packages = ["//..."],
)

package_group(
    name = "microfrontend",
    packages = ["//tensorflow/lite/experimental/microfrontend/..."],
)

package_group(
    name = "kernel_test_friends",
    packages = tflm_kernel_friends(),
)

tflm_cc_library(
    name = "micro_test",
    hdrs = [
        "micro_test.h",
    ],
    visibility = [
        ":kernel_test_friends",
        ":tflite_micro",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:system_setup",
        "//tensorflow/lite/micro:test_helpers",
    ],
)

tflm_cc_test(
    name = "util_test",
    srcs = [
        "util_test.cc",
    ],
    deps = [
        ":micro_test",
    ],
)

tflm_cc_library(
    name = "test_conv_model",
    srcs = [
        "test_conv_model.cc",
    ],
    hdrs = [
        "test_conv_model.h",
    ],
    visibility = [
        ":tflite_micro",
    ],
)

py_library(
    name = "generate_test_models_lib",
    srcs = ["generate_test_models.py"],
    visibility = [
        ":tflite_micro",
    ],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_binary(
    name = "generate_test_models",
    srcs = ["generate_test_models.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    tags = [
        "nomicro_static",  # TF dep incompatible w/ TF_LITE_STATIC_MEMORY.
        "noubsan",  # TODO(b/144512025): Fix raw_to_bitmap_test to fix ubsan failure.
    ],
    deps = [
        "@absl_py//absl:app",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)
