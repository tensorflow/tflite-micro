load(
    "//third_party/tflite_micro/tensorflow/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "micro_copts",
)

package(
    licenses = ["notice"],
)

package_group(
    name = "micro_top_level",
    packages = ["//third_party/tflite_micro/tensorflow/lite/micro"],
)

generate_cc_arrays(
    name = "generated_simple_add_model_cc",
    src = "models/simple_add_model.tflite",
    out = "models/simple_add_model_model_data.cc",
)

generate_cc_arrays(
    name = "generated_simple_add_model_hdr",
    src = "models/simple_add_model.tflite",
    out = "models/simple_add_model_model_data.h",
)

cc_library(
    name = "simple_add_model_data",
    srcs = [
        ":generated_simple_add_model_cc",
    ],
    hdrs = [
        ":generated_simple_add_model_hdr",
    ],
    copts = micro_copts(),
)

cc_binary(
    name = "baseline_memory_footprint",
    srcs = ["baseline_memory_footprint.cc"],
    deps = [
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tflite_micro/tensorflow/lite/micro:system_setup",
    ],
)

cc_binary(
    name = "interpreter_memory_footprint",
    srcs = ["interpreter_memory_footprint.cc"],
    deps = [
        ":simple_add_model_data",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_error_reporter",
        "//third_party/tflite_micro/tensorflow/lite/micro:micro_profiler",
        "//third_party/tflite_micro/tensorflow/lite/micro:op_resolvers",
        "//third_party/tflite_micro/tensorflow/lite/micro:system_setup",
        "//third_party/tflite_micro/tensorflow/lite/micro/benchmarks:micro_benchmark",
    ],
)
