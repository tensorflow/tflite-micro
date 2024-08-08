load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library", "flatbuffer_py_library")
load("@rules_python//python:defs.bzl", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

flatbuffer_cc_library(
    name = "metadata_flatbuffer_cc",
    srcs = ["metadata.fbs"],
)

flatbuffer_py_library(
    name = "original_flatbuffer_py",
    srcs = ["original.fbs"],
)

flatbuffer_py_library(
    name = "metadata_flatbuffer_py",
    srcs = ["metadata.fbs"],
)

cc_test(
    name = "metadata_test_cc",
    srcs = ["metadata_test.cc"],
    deps = [
        "metadata_flatbuffer_cc",
        "//tensorflow/lite/micro:hexdump",
        "@flatbuffers//:runtime_cc",
    ],
    size = "small",
)

py_binary(
    name = "compress",
    srcs = ["compress.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@flatbuffers//:runtime_py",
        "metadata_flatbuffer_py",
        "//tensorflow/lite/python:schema_py",
        requirement("bitarray"),
        requirement("numpy"),
        requirement("scikit-learn"),
    ],
)

py_binary(
    name = "view",
    srcs = [
        "view.py",
    ],
    deps = [
        "metadata_flatbuffer_py",
        "//tensorflow/lite/python:schema_py",
    ],
)

py_test(
    name = "metadata_test_py",
    main = "metadata_test.py",
    srcs = ["metadata_test.py"],
    deps = [
        "metadata_flatbuffer_py",
        "@flatbuffers//:runtime_py",
        requirement("hexdump"),
    ],
    size = "small",
)

py_test(
    name = "original_test_py",
    main = "original_test.py",
    srcs = ["original_test.py"],
    deps = [
        "original_flatbuffer_py",
        "@flatbuffers//:runtime_py",
        requirement("hexdump"),
    ],
    size = "small",
)

genrule(
    name = "hello_world_int8.compressed",
    srcs = ["//tensorflow/lite/micro/examples/hello_world/models:hello_world_int8.tflite"],
    outs = ["hello_world_int8.compressed.tflite"],
    cmd = "$(location :compress) --input_model_path $< --output_model_path $@",
    tools = [":compress"],
)
