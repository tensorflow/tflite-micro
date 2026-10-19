load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "cmsis_nn",
    srcs = glob(
        [
            "Source/**/*.c",
            "Source/**/*.h",
        ],
        exclude = [
            "Source/**/*_f16.*",
        ],
    ),
    hdrs = glob(["Include/**/*.h"]),
    copts = ["-Wno-unknown-attributes"],
    defines = [
        "CMSIS_NN_USE_SINGLE_ROUNDING=1",
        "ARM_NN_ENABLE_F32=1",
    ],
    includes = [
        ".",
        "Include",
    ],
)
