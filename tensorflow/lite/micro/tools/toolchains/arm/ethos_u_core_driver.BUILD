load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "ethos_u_core_driver",
    srcs = [
        "src/ethosu_device_u55_u65.c",
        "src/ethosu_driver.c",
        "src/ethosu_pmu.c",
    ],
    hdrs = glob([
        "include/*.h",
        "src/*.h",
    ]),
    copts = [
        "-Wno-unknown-attributes",
        "-Wno-pointer-to-int-cast",
        "-Wno-int-to-pointer-cast",
        "-DETHOSU_ARCH=u55",
        "-DETHOSU55",
        "-DETHOSU_LOG_SEVERITY=ETHOSU_LOG_WARN",
    ],
    includes = [
        "include",
        "src",
    ],
    target_compatible_with = [
        "@platforms//os:none",
        "@platforms//cpu:arm",
    ],
    deps = [
        "@cmsis_core",
    ],
)
