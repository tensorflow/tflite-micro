package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# The name schema_fbs is unchanged from upstream TF so that sync'ing shared
# TfLite/TFLM code does not require a change in the name for this BUILD target.
# For upstream TFL code, schema_fbs is a flatbuffer_cc_library whereas it is a
# standard cc_library (with the generated schems header) in the TFLM code.
cc_library(
    name = "schema_fbs",
    hdrs = ["schema_generated.h"],
    deps = [
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "schema_utils",
    srcs = ["schema_utils.cc"],
    hdrs = ["schema_utils.h"],
    deps = [
        ":schema_fbs",
        "//tensorflow/lite/kernels/internal:compatibility",
        "@flatbuffers//:runtime_cc",
    ],
)
