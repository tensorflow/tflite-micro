load("//codegen:build_def.bzl", "tflm_inference_library")

package(default_visibility = ["//visibility:public"])

tflm_inference_library(
    name = "hello_world_model",
    tflite_model = "//tensorflow/lite/micro/examples/hello_world/models:hello_world_int8.tflite",
)

cc_binary(
    name = "hello_world",
    srcs = ["hello_world.cc"],
    deps = [
        ":hello_world_model",
        "//tensorflow/lite/c:c_api_types",
    ],
)
