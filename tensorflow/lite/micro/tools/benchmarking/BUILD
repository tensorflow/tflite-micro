load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_binary",
    "tflm_cc_library",
    "tflm_defines",
)

tflm_cc_library(
    name = "op_resolver",
    hdrs = ["op_resolver.h"],
    visibility = ["//tensorflow/lite/micro/tools:__subpackages__"],
    deps = ["//tensorflow/lite/micro:op_resolvers"],
)

tflm_cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_profiler",
        "//tensorflow/lite/micro:recording_allocators",
    ],
)

tflm_cc_library(
    name = "generic_benchmark_lib",
    srcs = ["generic_model_benchmark.cc"],
    hdrs = ["show_meta_data.h"],
    defines = tflm_defines() + ["GENERIC_BENCHMARK_NO_META_DATA"],
    deps = [
        ":metrics",
        ":op_resolver",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_context",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_profiler",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro:system_setup",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

tflm_cc_binary(
    name = "tflm_benchmark",
    deps = [":generic_benchmark_lib"],
)
