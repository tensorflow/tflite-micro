load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

cc_library(
    name = "ibuffer_allocator",
    hdrs = [
        "ibuffer_allocator.h",
    ],
    copts = micro_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
    ],
)

cc_library(
    name = "non_persistent_arena_buffer_allocator",
    srcs = ["non_persistent_arena_buffer_allocator.cc"],
    hdrs = ["non_persistent_arena_buffer_allocator.h"],
    copts = micro_copts(),
    deps = [
        ":ibuffer_allocator",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:memory_helpers",
        "//tensorflow/lite/micro:micro_arena_constants",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:micro_error_reporter",
    ],
)

cc_test(
    name = "non_persistent_arena_buffer_allocator_test",
    srcs = ["non_persistent_arena_buffer_allocator_test.cc"],
    deps = [
        ":non_persistent_arena_buffer_allocator",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "persistent_arena_buffer_allocator",
    srcs = ["persistent_arena_buffer_allocator.cc"],
    hdrs = ["persistent_arena_buffer_allocator.h"],
    copts = micro_copts(),
    deps = [
        ":ibuffer_allocator",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:memory_helpers",
        "//tensorflow/lite/micro:micro_arena_constants",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:micro_error_reporter",
    ],
)

cc_test(
    name = "persistent_arena_buffer_allocator_test",
    srcs = ["persistent_arena_buffer_allocator_test.cc"],
    deps = [
        ":persistent_arena_buffer_allocator",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "simple_memory_allocator",
    srcs = [
        "single_arena_buffer_allocator.cc",
    ],
    hdrs = [
        "single_arena_buffer_allocator.h",
    ],
    copts = micro_copts(),
    deps = [
        ":ibuffer_allocator",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:memory_helpers",
        "//tensorflow/lite/micro:micro_arena_constants",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:micro_error_reporter",
    ],
)

cc_test(
    name = "simple_memory_allocator_test",
    srcs = [
        "single_arena_buffer_allocator_test.cc",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "recording_simple_memory_allocator",
    srcs = [
        "recording_single_arena_buffer_allocator.cc",
    ],
    hdrs = [
        "recording_single_arena_buffer_allocator.h",
    ],
    copts = micro_copts(),
    deps = [
        ":simple_memory_allocator",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:micro_framework",
    ],
)

cc_test(
    name = "recording_simple_memory_allocator_test",
    srcs = [
        "recording_single_arena_buffer_allocator_test.cc",
    ],
    deps = [
        ":recording_simple_memory_allocator",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
