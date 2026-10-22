load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//tensorflow/lite/micro/tools/toolchains:env.bzl",
    "ARM_TOOLCHAIN_BIN_PREFIX",
    "ARM_TOOLCHAIN_FILES_PREFIX",
    "QEMU_ARM_DATA",
    "tflm_arm_cpu_constraint_setting",
    "tflm_cc_toolchain",
    "tflm_sh_runner",
)
load(":cortex_m_toolchain_config.bzl", "cortex_m_toolchain_config")

package(default_visibility = ["//visibility:public"])

filegroup(name = "empty")

CPUS = {
    "m0": [
        "-mcpu=cortex-m0",
        "-mfloat-abi=soft",
        "-mthumb",
    ],
    "m4": [
        "-mcpu=cortex-m4",
        "-mfloat-abi=hard",
        "-mthumb",
        "-mfpu=fpv4-sp-d16",
    ],
    "m55": [
        "-mcpu=cortex-m55",
        "-mfpu=auto",
        "-mfloat-abi=hard",
        "-mthumb",
    ],
}

[
    cortex_m_toolchain_config(
        name = "%s_config" % name,
        ar = ARM_TOOLCHAIN_BIN_PREFIX + "arm-none-eabi-ar",
        c_compiler = ARM_TOOLCHAIN_BIN_PREFIX + "arm-none-eabi-gcc",
        copts = copts,
        cpp_compiler = ARM_TOOLCHAIN_BIN_PREFIX + "arm-none-eabi-g++",
        cpu = "cortex-%s" % name,
        strip = ARM_TOOLCHAIN_BIN_PREFIX + "arm-none-eabi-strip",
    )
    for name, copts in CPUS.items()
]

[
    tflm_cc_toolchain(
        name = "cc_toolchain_%s" % name,
        all_files = ARM_TOOLCHAIN_FILES_PREFIX + "everything",
        ar_files = ARM_TOOLCHAIN_FILES_PREFIX + "everything",
        as_files = ARM_TOOLCHAIN_FILES_PREFIX + "compiler_files",
        compiler_files = ARM_TOOLCHAIN_FILES_PREFIX + "compiler_files",
        dwp_files = ":empty",
        linker_files = ARM_TOOLCHAIN_FILES_PREFIX + "linker_files",
        objcopy_files = ":empty",
        strip_files = ARM_TOOLCHAIN_FILES_PREFIX + "everything",
        supports_param_files = 0,
        toolchain_config = ":%s_config" % name,
        toolchain_identifier = "cortex-%s-toolchain" % name,
    )
    for name in CPUS.keys()
]

ARM_CPU_CONSTRAINT_SETTING = tflm_arm_cpu_constraint_setting()

[
    constraint_value(
        name = name,
        constraint_setting = ARM_CPU_CONSTRAINT_SETTING,
    )
    for name in CPUS.keys()
]

[
    toolchain(
        name = "cortex_%s_toolchain" % name,
        exec_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:x86_64",
        ],
        target_compatible_with = [
            "@platforms//os:none",
            "@platforms//cpu:arm",
            ":%s" % name,
        ],
        toolchain = ":cc_toolchain_%s" % name,
        toolchain_type = "@rules_cc//cc:toolchain_type",
    )
    for name in CPUS.keys()
]

[
    platform(
        name = "%s_platform" % name,
        constraint_values = [
            "@platforms//os:none",
            "@platforms//cpu:arm",
            ":%s" % name,
        ],
    )
    for name in CPUS.keys()
]

config_setting(
    name = "cortex_m",
    constraint_values = [
        "@platforms//os:none",
        "@platforms//cpu:arm",
    ],
)

[
    tflm_sh_runner(
        name = "run_%s" % name,
        srcs = ["//tensorflow/lite/micro/tools/toolchains:run_qemu.sh"],
        data = QEMU_ARM_DATA,
    )
    for name in CPUS.keys()
]

[
    cc_test(
        name = "smoke_test_%s" % name,
        srcs = ["test_smoke.cc"],
        target_compatible_with = [
            "@platforms//os:none",
            "@platforms//cpu:arm",
            ":%s" % name,
        ],
    )
    for name in CPUS.keys()
]

alias(
    name = "smoke_test",
    actual = ":smoke_test_m55",
)
