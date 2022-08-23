load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

# This is the package group declaration to which targets for TensorFlow Lite
# Flatbuffer schema utilities.
#
# Its usage should be rare, and is often abused by tools that are doing
# Flatbuffer creation/manipulation in unofficially supported ways.

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "schema_fbs",
    hdrs = ["schema_generated.h"],
    deps = [
        "@flatbuffers//:runtime_cc",
    ],
)

flatbuffer_cc_library(
    name = "schema_fbs_generate",
    srcs = ["schema.fbs"],
)

cc_library(
    name = "schema_utils",
    srcs = ["schema_utils.cc"],
    hdrs = ["schema_utils.h"],
    deps = [
        ":schema_fbs",
        "//tensorflow/lite/kernels/internal:compatibility",
        "@flatbuffers//:runtime_cc",
    ],
)
