load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

# `bazel run` this target to generate compile_commands.json, which can be used
# by various tools like editors and LSPs to provide features like intelligent
# navigation and autocompletion based on the source graph and compiler commands.
refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = ["//..."],
)

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

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
