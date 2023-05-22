load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

#package(default_visibility = [
#    "//third_party/tflite_micro/google/benchmark:__subpackages__",
#])

vardef("CLIENT_TFLITE_FILE", "")

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        ":log_utils",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_profiler",
        "//third_party/tflite_micro/tensorflow/lite/micro:recording_allocators",
        "//third_party/tflite_micro/tensorflow/lite/micro/arena_allocator:recording_simple_memory_allocator",
    ],
)

generate_cc_arrays(
    name = "generated_model_cc",
    src = "$(CLIENT_TFLITE_FILE)",
    out = "benchmark.cc",
)

generate_cc_arrays(
    name = "generated_model_hdr",
    src = "$(CLIENT_TFLITE_FILE)",
    out = "benchmark.hdr",
)

cc_library(
    name = "log_utils",
    srcs = ["log_utils.cc"],
    hdrs = ["log_utils.h"],
    deps = [
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_log",
    ],
)

cc_library(
    name = "micro_benchmark",
    hdrs = ["micro_benchmark.h"],
    deps = [
        "//third_party/tensorflow/lite/c:c_api_types",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/schema:schema_fbs",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_log",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_profiler",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_profiler_interface",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_time",
        "//third_party/tflite_micro/tensorflow/lite/micro:op_resolvers",
        "//third_party/tflite_micro/tensorflow/lite/micro:recording_allocators",
    ],
)

cc_library(
    name = "generic_benchmark_lib",
    srcs = ["generic_model_benchmark.cc"],
    deps = [
        ":generated_model_cc",
        ":generated_model_hdr",
        ":log_utils",
        ":metrics",
        ":micro_benchmark",
        "//third_party/tensorflow/lite/c:c_api_types",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/schema:schema_fbs",
        "//third_party/tflite_micro/google/benchmark:op_resolver",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_framework",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_log",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_profiler",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_utils",
        "//third_party/tflite_micro/tensorflow/lite/micro:op_resolvers",
        "//third_party/tflite_micro/tensorflow/lite/micro:system_setup",
    ],
)

cc_binary(
    name = "tflm_benchmark",
    #    srcs = ["generic_model_benchmark.cc"],
    deps = [":generic_benchmark_lib"],
)
