load("//python:py_rules.bzl", "tflm_py_library")
load("//third_party/flatbuffers:build_defs.bzl", "flatbuffer_py_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["//tensorflow/lite/micro/schema:schema.fbs"],
)

tflm_py_library(
    name = "schema_util",
    srcs = ["schema_util.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//third_party/flatbuffers:runtime_py",
    ],
)
