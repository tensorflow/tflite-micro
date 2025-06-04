load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    licenses = ["notice"],
)

py_library(
    name = "add_four_numbers",
    srcs = ["add_four_numbers.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "resource_variables_lib",
    srcs = ["resource_variables_lib.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "resource_variables_test",
    srcs = ["resource_variables_test.py"],
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    deps = [
        ":resource_variables_lib",
        "//python/tflite_micro:runtime",
    ],
)
