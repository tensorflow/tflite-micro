load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
)

package(licenses = ["notice"])

tflm_cc_library(
    name = "register_signal_ops",
    srcs = [
        "delay.cc",
        "energy.cc",
        "fft_auto_scale_common.cc",
        "fft_auto_scale_kernel.cc",
        "filter_bank.cc",
        "filter_bank_log.cc",
        "filter_bank_spectral_subtraction.cc",
        "filter_bank_square_root.cc",
        "filter_bank_square_root_common.cc",
        "framer.cc",
        "irfft.cc",
        "overlap_add.cc",
        "pcan.cc",
        "rfft.cc",
        "stacker.cc",
        "window.cc",
    ],
    hdrs = [
        "fft_auto_scale_kernel.h",
        "filter_bank_square_root.h",
        "irfft.h",
        "rfft.h",
    ],
    visibility = [
        "//tensorflow/lite/micro",
    ],
    deps = [
        "//signal/src:circular_buffer",
        "//signal/src:energy",
        "//signal/src:fft_auto_scale",
        "//signal/src:filter_bank",
        "//signal/src:filter_bank_log",
        "//signal/src:filter_bank_spectral_subtraction",
        "//signal/src:filter_bank_square_root",
        "//signal/src:irfft",
        "//signal/src:overlap_add",
        "//signal/src:pcan_argc_fixed",
        "//signal/src:rfft",
        "//signal/src:window",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
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

cc_library(
    name = "stacker_flexbuffers_generated_data",
    srcs = [
        "stacker_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "stacker_flexbuffers_generated_data.h",
    ],
)

cc_test(
    name = "stacker_test",
    srcs = [
        "stacker_test.cc",
    ],
    deps = [
        ":register_signal_ops",
        ":stacker_flexbuffers_generated_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "energy_flexbuffers_generated_data",
    srcs = [
        "energy_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "energy_flexbuffers_generated_data.h",
    ],
)

cc_test(
    name = "energy_test",
    srcs = [
        "energy_test.cc",
    ],
    deps = [
        ":energy_flexbuffers_generated_data",
        ":register_signal_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "filter_bank_flexbuffers_generated_data",
    srcs = [
        "filter_bank_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "filter_bank_flexbuffers_generated_data.h",
    ],
)

cc_library(
    name = "filter_bank_log_flexbuffers_generated_data",
    srcs = [
        "filter_bank_log_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "filter_bank_log_flexbuffers_generated_data.h",
    ],
)

cc_library(
    name = "filter_bank_spectral_subtraction_flexbuffers_generated_data",
    srcs = [
        "filter_bank_spectral_subtraction_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "filter_bank_spectral_subtraction_flexbuffers_generated_data.h",
    ],
)

cc_test(
    name = "filter_bank_test",
    srcs = [
        "filter_bank_test.cc",
    ],
    deps = [
        ":filter_bank_flexbuffers_generated_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "filter_bank_log_test",
    srcs = [
        "filter_bank_log_test.cc",
    ],
    deps = [
        ":filter_bank_log_flexbuffers_generated_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "filter_bank_spectral_subtraction_test",
    srcs = [
        "filter_bank_spectral_subtraction_test.cc",
    ],
    deps = [
        ":filter_bank_spectral_subtraction_flexbuffers_generated_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "filter_bank_square_root_test",
    srcs = [
        "filter_bank_square_root_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "pcan_flexbuffers_generated_data",
    srcs = [
        "pcan_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "pcan_flexbuffers_generated_data.h",
    ],
)

cc_test(
    name = "pcan_test",
    srcs = [
        "pcan_test.cc",
    ],
    deps = [
        ":pcan_flexbuffers_generated_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels:kernel_runner",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
