load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# We are still generating and checking in the python bindings in the
# nightly sync to keep it working with the Makefiles.

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["//tensorflow/lite/schema:schema.fbs"],
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
