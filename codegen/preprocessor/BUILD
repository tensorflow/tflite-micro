load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library", "flatbuffer_py_library")

package(default_visibility = ["//visibility:public"])

flatbuffer_cc_library(
    name = "preprocessor_schema_fbs",
    srcs = ["preprocessor_schema.fbs"],
)

flatbuffer_py_library(
    name = "preprocessor_schema_py",
    srcs = ["preprocessor_schema.fbs"],
)
