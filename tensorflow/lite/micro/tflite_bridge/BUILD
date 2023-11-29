load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

cc_library(
    name = "flatbuffer_conversions_bridge",
    srcs = [
        "flatbuffer_conversions_bridge.cc",
    ],
    hdrs = [
        "flatbuffer_conversions_bridge.h",
    ],
    copts = micro_copts(),
    visibility = [
        "//tensorflow/lite/micro:__pkg__",
    ],
    deps = [
        ":micro_error_reporter",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "micro_error_reporter",
    srcs = [
        "micro_error_reporter.cc",
    ],
    hdrs = [
        "micro_error_reporter.h",
    ],
    copts = micro_copts(),
    deps = [
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:micro_log",
    ],
)
