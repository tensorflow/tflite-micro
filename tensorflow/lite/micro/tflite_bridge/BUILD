load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
)

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "flatbuffer_conversions_bridge",
    srcs = [
        "flatbuffer_conversions_bridge.cc",
    ],
    hdrs = [
        "flatbuffer_conversions_bridge.h",
    ],
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

tflm_cc_library(
    name = "micro_error_reporter",
    srcs = [
        "micro_error_reporter.cc",
    ],
    hdrs = [
        "micro_error_reporter.h",
    ],
    deps = [
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:micro_log",
    ],
)
