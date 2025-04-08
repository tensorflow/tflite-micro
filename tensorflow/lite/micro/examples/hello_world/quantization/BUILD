load("@rules_python//python:defs.bzl", "py_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "ptq",
    srcs = ["ptq.py"],
    data = ["//tensorflow/lite/micro/examples/hello_world/models:hello_world_float.tflite"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        requirement("numpy"),
        requirement("tensorflow"),
        "//python/tflite_micro:runtime",
    ],
)
