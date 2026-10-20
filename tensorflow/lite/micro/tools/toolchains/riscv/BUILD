load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc/toolchains:cc_toolchain.bzl", "cc_toolchain")
load(":riscv_toolchain_config.bzl", "riscv_toolchain_config")

package(default_visibility = ["//visibility:public"])

filegroup(name = "empty")

riscv_toolchain_config(
    name = "riscv32_config",
    ar = "@riscv_toolchain//:bin/riscv64-unknown-elf-ar",
    c_compiler = "@riscv_toolchain//:bin/riscv64-unknown-elf-gcc",
    cpp_compiler = "@riscv_toolchain//:bin/riscv64-unknown-elf-g++",
    cpu = "riscv32",
    strip = "@riscv_toolchain//:bin/riscv64-unknown-elf-strip",
)

cc_toolchain(
    name = "cc_toolchain_riscv32",
    all_files = "@riscv_toolchain//:everything",
    ar_files = "@riscv_toolchain//:everything",
    as_files = "@riscv_toolchain//:compiler_files",
    compiler_files = "@riscv_toolchain//:compiler_files",
    dwp_files = ":empty",
    linker_files = "@riscv_toolchain//:linker_files",
    objcopy_files = ":empty",
    strip_files = "@riscv_toolchain//:everything",
    supports_param_files = 0,
    toolchain_config = ":riscv32_config",
    toolchain_identifier = "riscv32-toolchain",
)

toolchain(
    name = "riscv32_toolchain",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:none",
        "@platforms//cpu:riscv32",
    ],
    toolchain = ":cc_toolchain_riscv32",
    toolchain_type = "@rules_cc//cc:toolchain_type",
)

platform(
    name = "riscv32_platform",
    constraint_values = [
        "@platforms//os:none",
        "@platforms//cpu:riscv32",
    ],
)

cc_test(
    name = "smoke_test",
    srcs = ["test_smoke.cc"],
)
