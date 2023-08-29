load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    licenses = ["notice"],
)

py_library(
    name = "add_four_numbers",
    srcs = ["add_four_numbers.py"],
    srcs_version = "PY3",
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
    ],
)

py_library(
    name = "resource_variables_lib",
    srcs = ["resource_variables_lib.py"],
    srcs_version = "PY3",
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "resource_variables_test",
    srcs = ["resource_variables_test.py"],
    srcs_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    deps = [
        ":resource_variables_lib",
        # TODO(b/286456378): update tflm_runtime to runtime when we are ready to
        # remove the alias.
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
    ],
)
