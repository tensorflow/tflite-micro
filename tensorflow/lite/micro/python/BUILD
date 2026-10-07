load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_rules.bzl", "tflm_py_library")
load("//third_party/flatbuffers:build_defs.bzl", "flatbuffer_py_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["//tensorflow/lite/schema:schema.fbs"],
)

tflm_py_library(
    name = "schema_util",
    srcs = ["schema_util.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("flatbuffers"),
    ],
)
