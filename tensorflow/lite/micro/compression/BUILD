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
