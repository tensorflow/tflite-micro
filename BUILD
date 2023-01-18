load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

# `bazel run` this target to generate compile_commands.json, which can be used
# by various tools like editors and LSPs to provide features like intelligent
# navigation and autocompletion based on the source graph and compiler commands.
refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = ["//..."],
)
