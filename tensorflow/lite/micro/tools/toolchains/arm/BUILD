load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc/toolchains:cc_toolchain.bzl", "cc_toolchain")
load(":cortex_m_toolchain_config.bzl", "cortex_m_toolchain_config")

package(default_visibility = ["//visibility:public"])

filegroup(name = "empty")

cc_library(
    name = "qemu_syscalls",
    srcs = ["qemu_syscalls.cc"],
    alwayslink = True,
)

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
        ar = "@arm_none_eabi//:bin/arm-none-eabi-ar",
        c_compiler = "@arm_none_eabi//:bin/arm-none-eabi-gcc",
        copts = copts,
        cpp_compiler = "@arm_none_eabi//:bin/arm-none-eabi-g++",
        cpu = "cortex-%s" % name,
    )
    for name, copts in CPUS.items()
]

[
    cc_toolchain(
        name = "cc_toolchain_%s" % name,
        all_files = "@arm_none_eabi//:everything",
        ar_files = "@arm_none_eabi//:everything",
        as_files = "@arm_none_eabi//:compiler_files",
        compiler_files = "@arm_none_eabi//:compiler_files",
        dwp_files = ":empty",
        linker_files = "@arm_none_eabi//:linker_files",
        objcopy_files = ":empty",
        strip_files = ":empty",
        supports_param_files = 0,
        toolchain_config = ":%s_config" % name,
        toolchain_identifier = "cortex-%s-toolchain" % name,
    )
    for name in CPUS.keys()
]

constraint_setting(name = "cortex_m_cpu")

[
    constraint_value(
        name = name,
        constraint_setting = ":cortex_m_cpu",
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

cc_test(
    name = "smoke_test",
    srcs = ["test_smoke.cc"],
    deps = [":qemu_syscalls"],
)
