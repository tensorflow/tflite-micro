cc_library(
    name = "op_resolver",
    hdrs = ["op_resolver.h"],
    deps = ["//tensorflow/lite/micro:op_resolvers"],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        ":log_utils",
        "//tensorflow/lite/micro:micro_profiler",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/arena_allocator:recording_simple_memory_allocator",
    ],
)

cc_library(
    name = "log_utils",
    srcs = ["log_utils.cc"],
    hdrs = ["log_utils.h"],
    deps = [
        "//tensorflow/lite/micro:micro_log",
    ],
)

cc_library(
    name = "generic_benchmark_lib",
    srcs = ["generic_model_benchmark.cc"],
    deps = [
        ":log_utils",
        ":metrics",
        ":op_resolver",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_profiler",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro:system_setup",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "tflm_benchmark",
    deps = [":generic_benchmark_lib"],
)
