# Description:
#   generated micro mutable op resolver test for a given model
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "micro_copts",
)

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)


generate_cc_arrays(name = "generated_${target}_model_data_cc",src = "${target}.tflite",out = "${target}_model_data.cc",)
generate_cc_arrays(name = "generated_${target}_model_data_hdr",src = "${target}.tflite",out = "${target}_model_data.h",)

% if verify_output:
generate_cc_arrays(
   name = "generated_${target}_input_${input_dtype}_test_data_cc",
   src = "${target}_input0_${input_dtype}.csv",
   out = "${target}_input_${input_dtype}_test_data.cc",
)

generate_cc_arrays(
   name = "generated_${target}_input_${input_dtype}_test_data_hdr",
   src = "${target}_input0_${input_dtype}.csv",
   out = "${target}_input_${input_dtype}_test_data.h",
)
% endif

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

cc_library(
    name = "models_and_testdata",
    srcs = [
        "generated_${target}_model_data_cc",
% if verify_output:
        "generated_${target}_input_${input_dtype}_test_data_cc",
        "generated_${target}_golden_${output_dtype}_test_data_cc",
% endif
    ],
    hdrs = [
        "generated_${target}_model_data_hdr",
% if verify_output:
        "generated_${target}_input_${input_dtype}_test_data_hdr",
        "generated_${target}_golden_${output_dtype}_test_data_hdr",
% endif
    ],
    copts = micro_copts(),
)

cc_library(
    name = "gen_micro_op_resolver",
    hdrs = ["gen_micro_mutable_op_resolver.h",],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "micro_mutable_op_resolver_test",
    srcs = [
        "micro_mutable_op_resolver_test.cc",
    ],
    copts = micro_copts(),
    deps = [
        ":gen_micro_op_resolver",
        ":models_and_testdata",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_resource_variable",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
