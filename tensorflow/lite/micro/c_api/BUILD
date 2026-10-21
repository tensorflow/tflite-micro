load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow/lite/micro:build_def.bzl", "tflm_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "c_api_lib",
    srcs = ["tflm_c_api.cc"],
    hdrs = ["tflm_c_api.h"],
    alwayslink = True,
    deps = [
        "//python/tflite_micro:compression_utils",
        "//python/tflite_micro:python_ops_resolver",
        "//python/tflite_micro:shared_library",
        "//tensorflow/lite/micro:micro_allocator",
        "//tensorflow/lite/micro:micro_common",
        "//tensorflow/lite/micro:micro_context",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_resource_variable",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/schema:schema_fbs",
        "@flatbuffers//:flatbuffers",
    ],
)

cc_binary(
    name = "libtflm.so",
    linkshared = True,
    linkstatic = True,
    deps = [
        ":c_api_lib",
    ],
)
