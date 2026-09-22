load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_rules.bzl", "tflm_py_binary")

tflm_py_binary(
    name = "ptq",
    srcs = ["ptq.py"],
    data = ["//tensorflow/lite/micro/examples/hello_world/models:hello_world_float.tflite"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
        "//python/tflite_micro:runtime",
    ],
)
