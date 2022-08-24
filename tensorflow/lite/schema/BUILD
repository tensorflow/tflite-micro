load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# We are still using a checked in version of the C++ flatbuffer bindings
# for the bazel build to keep it consistent with the Makefile build. And
# the bazel target is called schema_fbs so that the shared TfLite/TFLM code
# does not need any change.
#
# We have a second schema_fbs_generate bazel target so that TFLM can create
# its own version of the bindings from schema.fbs. This is mostly used as part
# of the nightly sync from upstream TF.

# TODO(b/243587789): Update so we only need one schema_fbs target (which will
# generate the schemas). May need to update some dependencies in BUILD files.

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
