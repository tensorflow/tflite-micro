load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")
load("@tflm_pip_deps//:requirements.bzl", "requirement")


flatbuffer_py_library(
    name = "schema_py_generate",
    srcs = ["//tensorflow/lite/schema:schema.fbs"],
)

py_library(
    name = "schema_py",
    srcs = ["schema_py_generated.py"],
    srcs_version = "PY3",
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("flatbuffers"),
    ],
)


py_library(
    name = "schema_util",
    srcs = ["schema_util.py"],
    srcs_version = "PY3",
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("flatbuffers"),
        requirement("tensorflow-cpu"),
    ],
)
