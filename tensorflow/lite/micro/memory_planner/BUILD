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
    name = "micro_memory_planner",
    hdrs = [
        "micro_memory_planner.h",
    ],
    copts = micro_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
    ],
)

cc_library(
    name = "linear_memory_planner",
    srcs = [
        "linear_memory_planner.cc",
    ],
    hdrs = [
        "linear_memory_planner.h",
    ],
    copts = micro_copts(),
    deps = [
        ":micro_memory_planner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:micro_error_reporter",
    ],
)

cc_library(
    name = "greedy_memory_planner",
    srcs = [
        "greedy_memory_planner.cc",
    ],
    hdrs = [
        "greedy_memory_planner.h",
    ],
    copts = micro_copts(),
    deps = [
        ":micro_memory_planner",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_string",
    ],
)

cc_test(
    name = "linear_memory_planner_test",
    srcs = [
        "linear_memory_planner_test.cc",
    ],
    deps = [
        ":linear_memory_planner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "greedy_memory_planner_test",
    srcs = [
        "greedy_memory_planner_test.cc",
    ],
    deps = [
        ":greedy_memory_planner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "memory_plan_struct",
    hdrs = [
        "memory_plan_struct.h",
    ],
    copts = micro_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:micro_utils",
    ],
)

cc_library(
    name = "non_persistent_buffer_planner_shim",
    srcs = ["non_persistent_buffer_planner_shim.cc"],
    hdrs = [
        "non_persistent_buffer_planner_shim.h",
    ],
    copts = micro_copts(),
    deps = [
        ":memory_plan_struct",
        ":micro_memory_planner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:micro_error_reporter",
    ],
)

cc_test(
    name = "non_persistent_buffer_planner_shim_test",
    srcs = [
        "non_persistent_buffer_planner_shim_test.cc",
    ],
    deps = [
        ":non_persistent_buffer_planner_shim",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
