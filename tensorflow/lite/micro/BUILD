load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_cc_test",
    "tflm_copts",
)

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-layering_check",  # buildozer: disable=no-layering-check, TODO(b/177257333): consider enabling layering check
        "-parse_headers",  # buildozer: disable=no-parse-headers, paser_headers is unavailable with bazel (http://b/175817117#comment4)
    ],
    licenses = ["notice"],
)

package_group(
    name = "micro",
    packages = ["//tensorflow/lite/micro/..."],
)

tflm_cc_library(
    name = "micro_compatibility",
    hdrs = [
        "compatibility.h",
    ],
)

tflm_cc_library(
    name = "compression",
    hdrs = [
        "compression.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
    ],
)

tflm_cc_library(
    # TODO(b/187093492): Rename to micro_interpreter.
    name = "micro_framework",
    srcs = [
        "micro_interpreter.cc",
    ],
    hdrs = [
        "micro_interpreter.h",
    ],
    deps = [
        ":memory_helpers",
        ":micro_allocator",
        ":micro_interpreter_context",
        ":micro_interpreter_graph",
        ":micro_profiler_interface",
        ":op_resolvers",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro/tflite_bridge:flatbuffer_conversions_bridge",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

tflm_cc_library(
    name = "micro_context",
    srcs = [
        "micro_context.cc",
    ],
    hdrs = [
        "micro_context.h",
    ],
    deps = [
        ":compression",
        ":micro_common",
        ":micro_graph",
        ":micro_log",
        ":micro_profiler",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro/kernels:decompress",
    ],
)

tflm_cc_library(
    name = "micro_interpreter_context",
    srcs = [
        "micro_interpreter_context.cc",
    ],
    hdrs = [
        "micro_interpreter_context.h",
    ],
    deps = [
        ":memory_helpers",
        ":micro_allocator",
        ":micro_context",
        ":micro_interpreter_graph",
        ":micro_log",
        ":micro_profiler_interface",
        "//tensorflow/lite/c:common",
    ],
)

tflm_cc_library(
    name = "micro_common",
    hdrs = [
        "micro_common.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
    ],
)

tflm_cc_library(
    name = "fake_micro_context",
    srcs = [
        "fake_micro_context.cc",
    ],
    hdrs = [
        "fake_micro_context.h",
    ],
    deps = [
        ":memory_helpers",
        ":micro_arena_constants",
        ":micro_context",
        ":micro_log",
        ":mock_micro_graph",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro/arena_allocator:simple_memory_allocator",
    ],
)

tflm_cc_library(
    name = "micro_graph",
    hdrs = ["micro_graph.h"],
    deps = [
        ":micro_common",
        ":micro_resource_variable",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

tflm_cc_library(
    name = "micro_interpreter_graph",
    srcs = ["micro_interpreter_graph.cc"],
    hdrs = ["micro_interpreter_graph.h"],
    deps = [
        ":memory_helpers",
        ":micro_allocator",
        ":micro_common",
        ":micro_context",
        ":micro_graph",
        ":micro_log",
        ":micro_profiler",
        ":micro_resource_variable",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

tflm_cc_library(
    name = "mock_micro_graph",
    srcs = ["mock_micro_graph.cc"],
    hdrs = ["mock_micro_graph.h"],
    deps = [
        ":micro_allocator",
        ":micro_graph",
        ":test_helpers",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

tflm_cc_library(
    name = "micro_allocator",
    srcs = [
        "micro_allocation_info.cc",
        "micro_allocator.cc",
    ],
    hdrs = [
        "micro_allocation_info.h",
        "micro_allocator.h",
    ],
    deps = [
        ":compression",
        ":flatbuffer_utils",
        ":memory_helpers",
        ":micro_arena_constants",
        ":micro_common",
        ":micro_compatibility",
        ":micro_log",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro/arena_allocator:ibuffer_allocator",
        "//tensorflow/lite/micro/arena_allocator:non_persistent_arena_buffer_allocator",
        "//tensorflow/lite/micro/arena_allocator:persistent_arena_buffer_allocator",
        "//tensorflow/lite/micro/arena_allocator:simple_memory_allocator",
        "//tensorflow/lite/micro/compression:metadata_saved",
        "//tensorflow/lite/micro/memory_planner:greedy_memory_planner",
        "//tensorflow/lite/micro/memory_planner:linear_memory_planner",
        "//tensorflow/lite/micro/memory_planner:micro_memory_planner",
        "//tensorflow/lite/micro/tflite_bridge:flatbuffer_conversions_bridge",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

tflm_cc_library(
    name = "micro_arena_constants",
    hdrs = [
        "micro_arena_constants.h",
    ],
    deps = [],
)

tflm_cc_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.cc"],
    hdrs = ["flatbuffer_utils.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

tflm_cc_library(
    name = "memory_helpers",
    srcs = ["memory_helpers.cc"],
    hdrs = ["memory_helpers.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:reference",
        "//tensorflow/lite/micro/tflite_bridge:flatbuffer_conversions_bridge",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

tflm_cc_library(
    name = "test_helpers",
    srcs = [
        "test_helper_custom_ops.cc",
        "test_helpers.cc",
    ],
    hdrs = [
        "test_helper_custom_ops.h",
        "test_helpers.h",
    ],
    deps = [
        ":compression",
        ":memory_helpers",
        ":micro_log",
        ":micro_utils",
        ":op_resolvers",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

tflm_cc_library(
    name = "op_resolvers",
    srcs = [
        "micro_op_resolver.cc",
    ],
    hdrs = [
        "micro_mutable_op_resolver.h",
        "micro_op_resolver.h",
    ],
    deps = [
        ":micro_compatibility",
        ":micro_log",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/micro/tflite_bridge:flatbuffer_conversions_bridge",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

tflm_cc_library(
    name = "debug_log",
    srcs = [
        "debug_log.cc",
    ],
    hdrs = [
        "debug_log.h",
    ],
)

tflm_cc_library(
    name = "micro_log",
    srcs = [
        "micro_log.cc",
    ],
    hdrs = [
        "micro_log.h",
    ],
    deps = [
        ":debug_log",
    ],
)

tflm_cc_library(
    name = "micro_resource_variable",
    srcs = [
        "micro_resource_variable.cc",
    ],
    hdrs = [
        "micro_resource_variable.h",
    ],
    deps = [
        ":micro_allocator",
        ":micro_log",
        ":micro_utils",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

tflm_cc_library(
    name = "micro_time",
    srcs = [
        "micro_time.cc",
    ],
    hdrs = [
        "micro_time.h",
    ],
    copts = tflm_copts() + ["-DTF_LITE_USE_CTIME"],
    deps = ["//tensorflow/lite/c:common"],
)

tflm_cc_library(
    name = "micro_profiler_interface",
    hdrs = [
        "micro_profiler_interface.h",
    ],
)

tflm_cc_library(
    name = "micro_profiler",
    srcs = [
        "micro_profiler.cc",
    ],
    hdrs = [
        "micro_profiler.h",
    ],
    deps = [
        ":micro_compatibility",
        ":micro_log",
        ":micro_profiler_interface",
        ":micro_time",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

tflm_cc_library(
    name = "micro_utils",
    srcs = [
        "micro_utils.cc",
    ],
    hdrs = [
        "micro_utils.h",
    ],
    deps = [
        ":memory_helpers",
        ":micro_log",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:op_macros",
    ],
)

tflm_cc_library(
    name = "recording_allocators",
    srcs = [
        "recording_micro_allocator.cc",
    ],
    hdrs = [
        "recording_micro_allocator.h",
        "recording_micro_interpreter.h",
    ],
    deps = [
        ":micro_allocator",
        ":micro_compatibility",
        ":micro_framework",
        ":micro_log",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro/arena_allocator:recording_simple_memory_allocator",
    ],
)

tflm_cc_library(
    name = "span",
    hdrs = ["span.h"],
)

tflm_cc_library(
    name = "static_vector",
    hdrs = ["static_vector.h"],
    deps = [
        "//tensorflow/lite/kernels:op_macros",
    ],
)

tflm_cc_library(
    name = "system_setup",
    srcs = [
        "system_setup.cc",
    ],
    hdrs = [
        "system_setup.h",
    ],
)

tflm_cc_library(
    name = "hexdump",
    srcs = [
        "hexdump.cc",
    ],
    hdrs = [
        "hexdump.h",
    ],
    deps = [
        ":micro_log",
        ":span",
        ":static_vector",
    ],
)

tflm_cc_test(
    name = "micro_log_test",
    srcs = [
        "micro_log_test.cc",
    ],
    deps = [
        ":micro_log",
        ":system_setup",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "micro_mutable_op_resolver_test",
    srcs = [
        "micro_mutable_op_resolver_test.cc",
    ],
    deps = [
        ":micro_framework",
        ":op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "micro_interpreter_context_test",
    srcs = [
        "micro_interpreter_context_test.cc",
    ],
    deps = [
        ":micro_allocator",
        ":micro_interpreter_context",
        ":micro_interpreter_graph",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "fake_micro_context_test",
    srcs = [
        "fake_micro_context_test.cc",
    ],
    deps = [
        ":fake_micro_context",
        ":micro_allocator",
        ":mock_micro_graph",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "micro_interpreter_test",
    srcs = [
        "micro_interpreter_test.cc",
    ],
    deps = [
        ":micro_compatibility",
        ":micro_framework",
        ":micro_profiler_interface",
        ":micro_utils",
        ":op_resolvers",
        ":recording_allocators",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "micro_allocator_test",
    srcs = [
        "micro_allocator_test.cc",
    ],
    deps = [
        ":memory_helpers",
        ":micro_allocator",
        ":micro_arena_constants",
        ":test_helpers",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro/memory_planner:memory_plan_struct",
        "//tensorflow/lite/micro/memory_planner:non_persistent_buffer_planner_shim",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/micro/testing:test_conv_model",
    ],
)

tflm_cc_test(
    name = "micro_allocation_info_test",
    srcs = [
        "micro_allocation_info_test.cc",
    ],
    deps = [
        ":micro_allocator",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "recording_micro_allocator_test",
    srcs = [
        "recording_micro_allocator_test.cc",
    ],
    deps = [
        ":micro_allocator",
        ":op_resolvers",
        ":recording_allocators",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/micro/testing:test_conv_model",
    ],
)

tflm_cc_test(
    name = "flatbuffer_utils_test",
    srcs = [
        "flatbuffer_utils_test.cc",
    ],
    tags = [
        "nomsan",  # TODO(b/192311485): See http://b/192311485#comment2
    ],
    deps = [
        ":flatbuffer_utils",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "memory_helpers_test",
    srcs = [
        "memory_helpers_test.cc",
    ],
    deps = [
        ":memory_helpers",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "span_test",
    size = "small",
    srcs = [
        "span_test.cc",
    ],
    deps = [
        ":span",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "testing_helpers_test",
    srcs = [
        "testing_helpers_test.cc",
    ],
    deps = [
        ":micro_framework",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "micro_utils_test",
    srcs = [
        "micro_utils_test.cc",
    ],
    deps = [
        ":micro_utils",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "micro_time_test",
    srcs = [
        "micro_time_test.cc",
    ],
    deps = [
        ":micro_time",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "micro_resource_variable_test",
    srcs = ["micro_resource_variable_test.cc"],
    deps = [
        ":micro_resource_variable",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "memory_arena_threshold_test",
    srcs = [
        "memory_arena_threshold_test.cc",
    ],
    deps = [
        ":op_resolvers",
        ":recording_allocators",
        "//tensorflow/lite/micro/benchmarks:keyword_scrambled_model_data",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/micro/testing:test_conv_model",
    ],
)

tflm_cc_test(
    name = "static_vector_test",
    size = "small",
    srcs = [
        "static_vector_test.cc",
    ],
    deps = [
        ":static_vector",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "hexdump_test",
    size = "small",
    srcs = [
        "hexdump_test.cc",
    ],
    deps = [
        ":hexdump",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

bzl_library(
    name = "build_def_bzl",
    srcs = ["build_def.bzl"],
    visibility = [":micro"],
)
