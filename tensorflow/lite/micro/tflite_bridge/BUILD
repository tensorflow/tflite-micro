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
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro/c:common",
        "//tensorflow/lite/micro/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)
