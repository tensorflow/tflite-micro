load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_python//python:defs.bzl", "py_library")

# `bazel run` this target to generate compile_commands.json, which can be used
# by various tools like editors and LSPs to provide features like intelligent
# navigation and autocompletion based on the source graph and compiler commands.
alias(
    name = "refresh_compile_commands",
    actual = "@wolfd_bazel_compile_commands//:generate_compile_commands",
)

bool_flag(
    name = "with_compression",
    build_setting_default = False,
)

config_setting(
    name = "with_compression_enabled",
    flag_values = {
        ":with_compression": "True",
    },
)

py_library(
    name = "tflite_micro_shim",
    srcs = ["tflite_micro.py"],
    visibility = ["//visibility:public"],
)
