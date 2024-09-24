load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# Note: when wanting to generate the schema_generated.h, you must build as:
# bazel build schema_fbs_srcs.
flatbuffer_cc_library(
    name = "schema_fbs",
    srcs = ["//tensorflow/compiler/mlir/lite/schema:schema.fbs"],
)

# Generic schema for inference on device (but with reflections makes bigger).
flatbuffer_cc_library(
    name = "schema_fbs_with_reflection",
    srcs = ["//tensorflow/compiler/mlir/lite/schema:schema.fbs"],
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
    hdrs = ["schema_utils.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
    ],
)
