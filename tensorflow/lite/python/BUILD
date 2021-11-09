load("@tflm_pip_deps//:requirements.bzl", "requirement")

# In upstream Tensorflow, this target is a flatbuffer_py_library which generates
# the Python bindings from schema.fbs
#
# We have made a decision to generate the Python bindings as part of the sync
# and directly commit schema_py_generated.py in the TFLM repository.
#
# The name of this BUILD target is kept the same as the upstream target to allow
# for fewer source transformations when going between the different
# repositories.
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
