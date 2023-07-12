load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)

package(licenses = ["notice"])

cc_library(
    name = "register_signal_ops",
    srcs = [
        "delay.cc",
        "framer.cc",
        "overlap_add.cc",
        "rfft.cc",
        "window.cc",
    ],
    hdrs = [
        "rfft.h",
    ],
    copts = micro_copts(),
    visibility = [
        "//tensorflow/lite/micro",
    ],
    deps = [
        "//signal/src:circular_buffer",
        "//signal/src:overlap_add",
        "//signal/src:rfft",
        "//signal/src:window",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:flatbuffer_utils",
        "//tensorflow/lite/micro:memory_helpers",
        "//tensorflow/lite/micro:micro_common",
        "//tensorflow/lite/micro:micro_context",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro/kernels:kernel_util",
    ],
)

cc_library(
    name = "fft_flexbuffers_generated_data",
    srcs = [
        "fft_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "fft_flexbuffers_generated_data.h",
    ],
)

cc_test(
    name = "fft_test",
    srcs = [
        "fft_test.cc",
    ],
    deps = [
        ":fft_flexbuffers_generated_data",
        ":register_signal_ops",
        "//signal/testdata:fft_test_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
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

cc_library(
    name = "framer_flexbuffers_generated_data",
    srcs = [
        "framer_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "framer_flexbuffers_generated_data.h",
    ],
)

cc_test(
    name = "framer_test",
    srcs = [
        "framer_test.cc",
    ],
    deps = [
        ":framer_flexbuffers_generated_data",
        ":register_signal_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "overlap_add_flexbuffers_generated_data",
    srcs = [
        "overlap_add_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "overlap_add_flexbuffers_generated_data.h",
    ],
)

cc_test(
    name = "overlap_add_test",
    srcs = [
        "overlap_add_test.cc",
    ],
    deps = [
        ":overlap_add_flexbuffers_generated_data",
        ":register_signal_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "delay_flexbuffers_generated_data",
    srcs = [
        "delay_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "delay_flexbuffers_generated_data.h",
    ],
)

cc_test(
    name = "delay_test",
    srcs = [
        "delay_test.cc",
    ],
    deps = [
        ":delay_flexbuffers_generated_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
