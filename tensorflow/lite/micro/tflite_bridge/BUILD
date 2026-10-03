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
    hdrs = [
        "flatbuffer_conversions_bridge.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/lite/micro:flatbuffer_conversions",
    ],
)
