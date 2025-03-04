# Description:
#   generated integration test for one specific kernel in a model.
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "tflm_cc_library",
    "tflm_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

generate_cc_arrays(
    name = "generated_quantize0_model_data_cc",
    src = "quantize0.tflite",
    out = "quantize0_model_data.cc",
)

generate_cc_arrays(
    name = "generated_quantize0_model_data_hdr",
    src = "quantize0.tflite",
    out = "quantize0_model_data.h",
)

generate_cc_arrays(
    name = "generated_quantize1_model_data_cc",
    src = "quantize1.tflite",
    out = "quantize1_model_data.cc",
)

generate_cc_arrays(
    name = "generated_quantize1_model_data_hdr",
    src = "quantize1.tflite",
    out = "quantize1_model_data.h",
)

generate_cc_arrays(
    name = "generated_quantize0_input0_int32_test_data_cc",
    src = "quantize0_input0_int32.csv",
    out = "quantize0_input0_int32_test_data.cc",
)

generate_cc_arrays(
    name = "generated_quantize0_input0_int32_test_data_hdr",
    src = "quantize0_input0_int32.csv",
    out = "quantize0_input0_int32_test_data.h",
)

generate_cc_arrays(
    name = "generated_quantize0_golden_int16_test_data_cc",
    src = "quantize0_golden_int16.csv",
    out = "quantize0_golden_int16_test_data.cc",
)

generate_cc_arrays(
    name = "generated_quantize0_golden_int16_test_data_hdr",
    src = "quantize0_golden_int16.csv",
    out = "quantize0_golden_int16_test_data.h",
)

generate_cc_arrays(
    name = "generated_quantize1_input0_int16_test_data_cc",
    src = "quantize1_input0_int16.csv",
    out = "quantize1_input0_int16_test_data.cc",
)

generate_cc_arrays(
    name = "generated_quantize1_input0_int16_test_data_hdr",
    src = "quantize1_input0_int16.csv",
    out = "quantize1_input0_int16_test_data.h",
)

generate_cc_arrays(
    name = "generated_quantize1_golden_int32_test_data_cc",
    src = "quantize1_golden_int32.csv",
    out = "quantize1_golden_int32_test_data.cc",
)

generate_cc_arrays(
    name = "generated_quantize1_golden_int32_test_data_hdr",
    src = "quantize1_golden_int32.csv",
    out = "quantize1_golden_int32_test_data.h",
)

tflm_cc_library(
    name = "models_and_testdata",
    srcs = [
        "generated_quantize0_golden_int16_test_data_cc",
        "generated_quantize0_input0_int32_test_data_cc",
        "generated_quantize0_model_data_cc",
        "generated_quantize1_golden_int32_test_data_cc",
        "generated_quantize1_input0_int16_test_data_cc",
        "generated_quantize1_model_data_cc",
    ],
    hdrs = [
        "generated_quantize0_golden_int16_test_data_hdr",
        "generated_quantize0_input0_int32_test_data_hdr",
        "generated_quantize0_model_data_hdr",
        "generated_quantize1_golden_int32_test_data_hdr",
        "generated_quantize1_input0_int16_test_data_hdr",
        "generated_quantize1_model_data_hdr",
    ],
)

tflm_cc_test(
    name = "integration_test",
    srcs = [
        "integration_tests.cc",
    ],
    deps = [
        ":models_and_testdata",
        "//python/tflite_micro:python_ops_resolver",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_resource_variable",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
