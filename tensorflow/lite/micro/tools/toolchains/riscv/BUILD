load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//tensorflow/lite/micro/tools/toolchains:env.bzl",
    "CC_TOOLCHAIN_TYPE",
    "QEMU_RISCV32_DATA",
    "RISCV_LINKER_FILES",
    "RISCV_TOOLCHAIN_BIN_PREFIX",
    "RISCV_TOOLCHAIN_FILES_PREFIX",
    "tflm_cc_toolchain",
    "tflm_sh_runner",
)
load(
    "//tensorflow/lite/micro/tools/toolchains:gcc_toolchain_config.bzl",
    "gcc_toolchain_config",
)
load(
    "//tensorflow/lite/micro/tools/toolchains:platform_suites.bzl",
    "tflm_platform_sh_test",
    "tflm_platform_target",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(name = "empty")

gcc_toolchain_config(
    name = "riscv32_config",
    ar = RISCV_TOOLCHAIN_BIN_PREFIX + "riscv64-unknown-elf-ar",
    arch_flags = [
        "-march=rv32imc",
        "-mabi=ilp32",
        "-mcmodel=medany",
        "-mexplicit-relocs",
    ],
    c_compiler = RISCV_TOOLCHAIN_BIN_PREFIX + "riscv64-unknown-elf-gcc",
    compile_flags = [
        "-fno-builtin-printf",
        "-DTF_LITE_MCU_DEBUG_LOG",
        "-DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS",
        "-funsigned-char",
        "-fno-delete-null-pointer-checks",
    ],
    cpp_compiler = RISCV_TOOLCHAIN_BIN_PREFIX + "riscv64-unknown-elf-g++",
    cpu = "riscv32",
    cxxopts = [
        "-fpermissive",
        "-fno-use-cxa-atexit",
        "-DTF_LITE_USE_GLOBAL_MIN",
        "-DTF_LITE_USE_GLOBAL_MAX",
    ],
    linkopts = [
        "--specs=nano.specs",
        "-u",
        "_printf_float",
        "-mno-relax",
    ],
    strip = RISCV_TOOLCHAIN_BIN_PREFIX + "riscv64-unknown-elf-strip",
    tags = ["manual"],
    target_system_name = "riscv32-unknown-elf",
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
    supports_param_files = False,
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
    toolchain_type = CC_TOOLCHAIN_TYPE,
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
    tags = [
        "manual",
        "notap",
    ],
    target_compatible_with = [
        "@platforms//os:none",
        "@platforms//cpu:riscv32",
    ],
)

tflm_platform_target(
    name = "smoke_test_riscv32",
    profile = "riscv32",
    target = ":smoke_test",
)

[
    cc_binary(
        name = "qemu_exit_%s_guest" % code,
        testonly = True,
        srcs = ["//tensorflow/lite/micro/tools/toolchains:qemu_exit_test_guest.cc"],
        defines = ["QEMU_TEST_EXIT_CODE=%s" % code],
        tags = ["manual"],
        target_compatible_with = [
            "@platforms//os:none",
            "@platforms//cpu:riscv32",
        ],
    )
    for code in [
        "0",
        "37",
    ]
]

[
    tflm_platform_target(
        name = "qemu_exit_%s_riscv32" % code,
        profile = "riscv32",
        target = ":qemu_exit_%s_guest" % code,
    )
    for code in [
        "0",
        "37",
    ]
]

tflm_platform_sh_test(
    name = "qemu_exit_code_test",
    srcs = ["//tensorflow/lite/micro/tools/toolchains:qemu_exit_code_test.sh"],
    args = [
        "$(rlocationpath :qemu_exit_0_riscv32)",
        "$(rlocationpath :qemu_exit_37_riscv32)",
    ],
    data = [
        ":qemu_exit_0_riscv32",
        ":qemu_exit_37_riscv32",
    ],
    profile = "riscv32",
)
