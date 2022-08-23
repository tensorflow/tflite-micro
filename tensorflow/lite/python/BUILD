load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

# We are still using a checked in version of the python flatbuffer bindings
# for the bazel build to keep it consistent with the Makefile build. And
# the bazel target is called schema_py so that the shared TfLite/TFLM code
# does not need any change.
#
# We have a second schema_py_generate bazel target so that TFLM can create
# its own version of the bindings from schema.fbs. This is mostly used as part
# of the nightly sync from upstream TF.

# b/243587789
# Update so we only need one schema_py target (which will generate
# the schemas). May need to update some dependencies in BUILD files.

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
