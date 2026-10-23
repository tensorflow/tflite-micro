load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//tensorflow/lite/micro/tools/toolchains:env.bzl",
    "CORSTONE300_FVP_DATA",
    "tflm_sh_runner",
)
load(
    "//tensorflow/lite/micro/tools/toolchains:platform_suites.bzl",
    "tflm_platform_target",
)

package(default_visibility = ["//visibility:public"])

exports_files(["corstone300.ld"])

tflm_sh_runner(
    name = "run_corstone300",
    srcs = ["run_corstone300.sh"],
    data = CORSTONE300_FVP_DATA,
)

cc_library(
    name = "corstone_bsp",
    srcs = [
        "corstone_startup.c",
        "corstone_syscalls.cc",
        "corstone_system.c",
    ],
    additional_linker_inputs = [":corstone300.ld"],
    linkopts = [
        "-Wl,-u,__VECTOR_TABLE",
        "-Wl,-u,__VECTOR_TABLE_SRAM",
        "-Wl,--entry=Reset_Handler",
        "--specs=nosys.specs",
        "-T$(execpath :corstone300.ld)",
    ],
    target_compatible_with = [
        "@platforms//os:none",
        "@platforms//cpu:arm",
    ],
    deps = [
        "@ethos_u_core_driver",
    ],
    alwayslink = True,
)

cc_test(
    name = "smoke_test",
    srcs = ["corstone_smoke_test.cc"],
    tags = [
        "manual",
        "notap",
    ],
    target_compatible_with = [
        "@platforms//os:none",
        "@platforms//cpu:arm",
    ],
    deps = [
        ":corstone_bsp",
    ],
)

tflm_platform_target(
    name = "smoke_test_corstone300",
    profile = "corstone300",
    target = ":smoke_test",
)
