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

% for target in targets:
generate_cc_arrays(name = "generated_${target}_model_data_cc",src = "${target}.tflite",out = "${target}_model_data.cc",)
generate_cc_arrays(name = "generated_${target}_model_data_hdr",src = "${target}.tflite",out = "${target}_model_data.h",)
% endfor

% for target in targets:
% for input_idx, input in enumerate(inputs):
generate_cc_arrays(
  name = "generated_${target}_input${input_idx}_${input_dtypes[input_idx]}_test_data_cc",
  src = "${target}_input${input_idx}_${input_dtypes[input_idx]}.csv",
  out = "${target}_input${input_idx}_${input_dtypes[input_idx]}_test_data.cc",
)

generate_cc_arrays(
  name = "generated_${target}_input${input_idx}_${input_dtypes[input_idx]}_test_data_hdr",
  src = "${target}_input${input_idx}_${input_dtypes[input_idx]}.csv",
  out = "${target}_input${input_idx}_${input_dtypes[input_idx]}_test_data.h",
)
% endfor

generate_cc_arrays(
  name = "generated_${target}_golden_${output_dtype}_test_data_cc",
  src = "${target}_golden_${output_dtype}.csv",
  out = "${target}_golden_${output_dtype}_test_data.cc",
)

generate_cc_arrays(
  name = "generated_${target}_golden_${output_dtype}_test_data_hdr",
  src = "${target}_golden_${output_dtype}.csv",
  out = "${target}_golden_${output_dtype}_test_data.h",
)
% endfor

tflm_cc_library(
    name = "models_and_testdata",
    srcs = [
% for target in targets:
        "generated_${target}_model_data_cc",
% for input_idx, input in enumerate(inputs):
        "generated_${target}_input${input_idx}_${input_dtypes[input_idx]}_test_data_cc",
% endfor
        "generated_${target}_golden_${output_dtype}_test_data_cc",
% endfor
    ],
    hdrs = [
% for target in targets:
        "generated_${target}_model_data_hdr",
% for input_idx, input in enumerate(inputs):
        "generated_${target}_input${input_idx}_${input_dtypes[input_idx]}_test_data_hdr",
% endfor
        "generated_${target}_golden_${output_dtype}_test_data_hdr",
% endfor
    ],
)

tflm_cc_test(
    name = "integration_test",
    srcs = [
        "integration_tests.cc",
    ],
    deps = [
        ":models_and_testdata",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_resource_variable",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//python/tflite_micro:python_ops_resolver",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
