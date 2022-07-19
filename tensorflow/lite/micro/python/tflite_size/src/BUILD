load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

# Append _lib at the end to avoid naming collision with the extension below
# because internal tool appends a _pybind suffix.
pybind_library(
    name = "flatbuffer_size_wrapper_lib",
    srcs = [
        "flatbuffer_size.cc",
        "flatbuffer_size_wrapper.cc",
    ],
    hdrs = [
        "flatbuffer_size.h",
        "flatbuffer_size_wrapper.h",
        "schema_generated_with_reflective_type.h",
    ],
    deps = [
        "@flatbuffers",
    ],
)

# pybind_extension() appends ".so" to "name" so the actual target name contains
# the ".so" suffix
pybind_extension(
    name = "flatbuffer_size_wrapper_pybind",
    srcs = [
        "flatbuffer_size_wrapper_pybind.cc",
    ],
    deps = [
        ":flatbuffer_size_wrapper_lib",
    ],
)

py_library(
    name = "flatbuffer_size_lib",
    srcs = [
        "flatbuffer_size_graph.py",
        "flatbuffer_size_graph_html_converter.py",
    ],
    data = [
        ":flatbuffer_size_wrapper_pybind.so",
    ],
    srcs_version = "PY3",
)

py_binary(
    name = "flatbuffer_size",
    srcs = [
        "flatbuffer_size.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":flatbuffer_size_lib",
    ],
)
