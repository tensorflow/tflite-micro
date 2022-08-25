load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# We are still generating and checking in the C++ bindings in the
# nightly sync to keep it working with the Makefiles.

flatbuffer_cc_library(
    name = "schema_fbs",
    srcs = ["schema.fbs"],
)

# Generic schema for inference on device (but with reflections makes bigger).
flatbuffer_cc_library(
    name = "schema_fbs_with_reflection",
    srcs = ["schema.fbs"],
    flatc_args = [
        "--reflect-types",
        "--reflect-names",
        "--no-union-value-namespacing",
        "--gen-object-api",
    ],
    out_prefix = "reflection/",
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
