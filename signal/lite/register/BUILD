load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//python:py_rules.bzl", "pybind_extension", "tflm_oss_manual_tags")

# These targets integrate TFLM signal custom ops with full TensorFlow Lite and
# LiteRT. They are marked "manual" in standalone TFLM builds because they
# require external @org_tensorflow (and @litert for signal_litert_test)
# dependencies.
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "signal_ops_registerer",
    srcs = ["signal_ops_registerer.cc"],
    hdrs = ["signal_ops_registerer.h"],
    tags = tflm_oss_manual_tags(),
    deps = [
        "//signal/lite/kernels:signal_ops",
        "@org_tensorflow//tensorflow/lite:mutable_op_resolver",
    ],
)

pybind_extension(
    name = "pywrap_signal_ops",  # :pywrap_signal_ops.so
    srcs = ["signal_ops_wrapper.cc"],
    tags = tflm_oss_manual_tags(),
    deps = [
        ":signal_ops_registerer",
        "@org_tensorflow//tensorflow/lite:mutable_op_resolver",
    ],
)

cc_test(
    name = "signal_ops_test",
    size = "small",
    srcs = ["signal_ops_test.cc"],
    tags = tflm_oss_manual_tags(),
    deps = [
        ":signal_ops_registerer",
        "//signal/lite/kernels:signal_ops",
        "//third_party/tflite_c_headers:common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/lite:mutable_op_resolver",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "signal_litert_test",
    size = "small",
    srcs = ["signal_litert_test.cc"],
    tags = tflm_oss_manual_tags(),
    deps = [
        "//signal/lite/kernels:signal_ops",
        "//third_party/tflite_c_headers:c_api_opaque",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@litert//litert/c/options:litert_cpu_options",
        "@litert//litert/cc:litert_api_types",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/tflite_support/custom_op",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)
