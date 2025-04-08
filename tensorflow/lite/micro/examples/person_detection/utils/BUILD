load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    features = ["-layering_check"],
    licenses = ["notice"],
)

py_binary(
    name = "raw_to_bitmap",
    srcs = ["raw_to_bitmap.py"],
    deps = [
        requirement("numpy"),
    ],
)

py_library(
    name = "raw_to_bitmap_lib",
    srcs = ["raw_to_bitmap.py"],
    deps = [
        requirement("numpy"),
    ],
)

py_test(
    name = "raw_to_bitmap_test",
    srcs = ["raw_to_bitmap_test.py"],
    data = glob(["testdata/**"]),
    tags = [
        "nomicro_static",  # TF dep incompatible w/ TF_LITE_STATIC_MEMORY.
        "notap",  # TODO(b/186679612)
        "noubsan",  # TODO(b/144512025): Fix raw_to_bitmap_test to fix ubsan failure.
    ],
    deps = [
        ":raw_to_bitmap_lib",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)
