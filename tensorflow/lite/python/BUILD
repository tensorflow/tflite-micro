load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["//tensorflow/compiler/mlir/lite/schema:schema.fbs"],
)

py_library(
    name = "schema_util",
    srcs = ["schema_util.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("flatbuffers"),
        requirement("tensorflow"),
    ],
)
