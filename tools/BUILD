load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("//python:py_rules.bzl", "tflm_py_test")

package(
    default_visibility = ["//visibility:public"],
)

py_binary(
    name = "expand_stamp_vars",
    srcs = ["expand_stamp_vars.py"],
)

tflm_py_test(
    name = "expand_stamp_vars_test",
    srcs = ["expand_stamp_vars_test.py"],
    deps = [
        ":expand_stamp_vars",
    ],
)
