load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//tensorflow/lite/micro/tools/toolchains:env.bzl",
    "CORSTONE300_FVP_DATA",
    "tflm_sh_runner",
)
load(
    "//tensorflow/lite/micro/tools/toolchains:platform_suites.bzl",
    "tflm_platform_sh_test",
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
        "-Wl,--entry=Reset_Handler",
        "-Wl,--orphan-handling=warn",
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

_FAULT_MODES = {
    "exit_42": "1",
    "stkof_fault": "3",
    "udf_fault": "2",
}

[
    cc_binary(
        name = "%s_guest" % mode_name,
        testonly = True,
        srcs = ["corstone_exit_test_guest.cc"],
        defines = ["CORSTONE_TEST_MODE=%s" % mode_id],
        tags = ["manual"],
        target_compatible_with = [
            "@platforms//os:none",
            "@platforms//cpu:arm",
        ],
        deps = [
            ":corstone_bsp",
        ],
    )
    for mode_name, mode_id in _FAULT_MODES.items()
]

[
    tflm_platform_target(
        name = "%s_corstone300" % mode_name,
        profile = "corstone300",
        target = ":%s_guest" % mode_name,
    )
    for mode_name in _FAULT_MODES.keys()
]

tflm_platform_sh_test(
    name = "corstone_exit_code_test",
    srcs = ["corstone_exit_code_test.sh"],
    args = [
        "$(rlocationpath :smoke_test_corstone300)",
        "$(rlocationpath :exit_42_corstone300)",
        "$(rlocationpath :udf_fault_corstone300)",
        "$(rlocationpath :stkof_fault_corstone300)",
    ],
    data = [
        ":exit_42_corstone300",
        ":smoke_test_corstone300",
        ":stkof_fault_corstone300",
        ":udf_fault_corstone300",
    ],
    profile = "corstone300",
)
