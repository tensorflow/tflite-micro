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
        "ARM_NN_ENABLE_F32=1",
    ] + select({
        "@tflite_micro//:with_single_rounding_enabled": ["CMSIS_NN_USE_SINGLE_ROUNDING=1"],
        "//conditions:default": [],
    }),
    includes = [
        ".",
        "Include",
    ],
)
