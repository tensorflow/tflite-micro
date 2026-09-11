load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_rules.bzl", "tflm_py_library", "tflm_py_test")
load("//tensorflow/lite/micro:build_def.bzl", "INCOMPATIBLE_WITH_WINDOWS")

package(
    licenses = ["notice"],
)

tflm_py_library(
    name = "add_four_numbers",
    srcs = ["add_four_numbers.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

tflm_py_library(
    name = "resource_variables_lib",
    srcs = ["resource_variables_lib.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

tflm_py_test(
    name = "resource_variables_test",
    srcs = ["resource_variables_test.py"],
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":resource_variables_lib",
        requirement("numpy"),
        "//python/tflite_micro:runtime",
    ],
)
