load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "cmsis_core",
    hdrs = glob(["CMSIS/Core/Include/**/*.h"]),
    includes = ["CMSIS/Core/Include"],
)
