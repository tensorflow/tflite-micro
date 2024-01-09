load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = [
        "hello_world_float.tflite",
        "hello_world_int8.tflite",
    ],
    visibility = [
        "//codegen/examples/hello_world:__subpackages__",
        "//tensorflow/lite/micro/examples/hello_world:__subpackages__",
    ],
)

generate_cc_arrays(
    name = "generated_hello_world_float_model_cc",
    src = "hello_world_float.tflite",
    out = "hello_world_float_model_data.cc",
)

generate_cc_arrays(
    name = "generated_hello_world_float_model_hdr",
    src = "hello_world_float.tflite",
    out = "hello_world_float_model_data.h",
)

generate_cc_arrays(
    name = "generated_hello_world_int8_model_cc",
    src = "hello_world_int8.tflite",
    out = "hello_world_int8_model_data.cc",
)

generate_cc_arrays(
    name = "generated_hello_world_int8_model_hdr",
    src = "hello_world_int8.tflite",
    out = "hello_world_int8_model_data.h",
)
