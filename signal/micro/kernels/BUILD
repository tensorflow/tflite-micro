load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)

package(
    licenses = ["notice"],
)


cc_library(
    name = "signal_ops_python",
    srcs = [
        "signal_ops_registerer_python.cc",
    ],
    hdrs = [
        "signal_ops_registerer_python.h",
    ],
    copts = micro_copts(),
    visibility = [
        "//tensorflow/lite/micro",
        "//signal:__subpackages__",
    ],
    deps = [
        ":register_signal_ops",
        "//tensorflow/lite/micro/python/interpreter/src:python_ops_resolver",
    ],
)

cc_library(
    name = "register_signal_ops",
    srcs = [
        "window.cc",
    ],
    hdrs = [
        "register_signal_ops.h",
    ],
    visibility = [
        "//tensorflow/lite/micro",
    ],
    copts = micro_copts(),
    deps = [
        "//tensorflow/lite/micro:micro_common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:flatbuffer_utils",
        "//tensorflow/lite/micro:memory_helpers",
        "//tensorflow/lite/micro:micro_context",
        "//tensorflow/lite/micro:micro_utils",
        "//signal/src:window",
        "//tensorflow/lite/micro/kernels:kernel_util",
    ],
)

cc_library(
    name = "window_flexbuffers_generated_data",
    srcs = [
        "window_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "window_flexbuffers_generated_data.h",
    ],
)

cc_test(
    name = "window_test",
    srcs = [
        "window_test.cc",
    ],
    deps = [
        ":register_signal_ops",
        ":window_flexbuffers_generated_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
