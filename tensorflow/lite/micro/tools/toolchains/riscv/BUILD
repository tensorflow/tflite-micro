load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//tensorflow/lite/micro/tools/toolchains:env.bzl",
    "QEMU_RISCV32_DATA",
    "RISCV_LINKER_FILES",
    "RISCV_TOOLCHAIN_BIN_PREFIX",
    "RISCV_TOOLCHAIN_FILES_PREFIX",
    "tflm_cc_toolchain",
    "tflm_sh_runner",
)
load(":riscv_toolchain_config.bzl", "riscv_toolchain_config")

package(default_visibility = ["//visibility:public"])

filegroup(name = "empty")

riscv_toolchain_config(
    name = "riscv32_config",
    ar = RISCV_TOOLCHAIN_BIN_PREFIX + "riscv64-unknown-elf-ar",
    c_compiler = RISCV_TOOLCHAIN_BIN_PREFIX + "riscv64-unknown-elf-gcc",
    cpp_compiler = RISCV_TOOLCHAIN_BIN_PREFIX + "riscv64-unknown-elf-g++",
    cpu = "riscv32",
    strip = RISCV_TOOLCHAIN_BIN_PREFIX + "riscv64-unknown-elf-strip",
)

tflm_cc_toolchain(
    name = "cc_toolchain_riscv32",
    all_files = RISCV_TOOLCHAIN_FILES_PREFIX + "everything",
    ar_files = RISCV_TOOLCHAIN_FILES_PREFIX + "everything",
    as_files = RISCV_TOOLCHAIN_FILES_PREFIX + "compiler_files",
    compiler_files = RISCV_TOOLCHAIN_FILES_PREFIX + "compiler_files",
    dwp_files = ":empty",
    linker_files = RISCV_LINKER_FILES,
    objcopy_files = ":empty",
    strip_files = RISCV_TOOLCHAIN_FILES_PREFIX + "everything",
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

cc_library(
    name = "riscv_bsp",
    srcs = ["riscv_bsp.c"],
    target_compatible_with = [
        "@platforms//os:none",
        "@platforms//cpu:riscv32",
    ],
    alwayslink = True,
)

tflm_sh_runner(
    name = "run_riscv32",
    srcs = ["//tensorflow/lite/micro/tools/toolchains:run_qemu.sh"],
    data = QEMU_RISCV32_DATA,
)

cc_test(
    name = "smoke_test",
    srcs = ["test_smoke.cc"],
)
