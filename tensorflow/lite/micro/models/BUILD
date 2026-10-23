# Description:
#   TensorFlow Lite for Microcontrollers Vision Example.
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "tflm_cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob(["*.tflite"]))

filegroup(
    name = "models",
    data = glob(["*.tflite"]),
)

generate_cc_arrays(
    name = "generated_person_detect_model_cc",
    src = "person_detect.tflite",
    out = "person_detect_model_data.cc",
)

generate_cc_arrays(
    name = "generated_person_detect_model_hdr",
    src = "person_detect.tflite",
    out = "person_detect_model_data.h",
)

generate_cc_arrays(
    name = "generated_person_detect_vela_model_cc",
    src = "person_detect_vela.tflite",
    out = "person_detect_vela_model_data.cc",
)

generate_cc_arrays(
    name = "generated_person_detect_vela_model_hdr",
    src = "person_detect_vela.tflite",
    out = "person_detect_vela_model_data.h",
)

tflm_cc_library(
    name = "person_detect_vela_model_data",
    srcs = [
        ":generated_person_detect_vela_model_cc",
    ],
    hdrs = [
        ":generated_person_detect_vela_model_hdr",
    ],
)

generate_cc_arrays(
    name = "generated_keyword_scrambled_model_hdr",
    src = "keyword_scrambled.tflite",
    out = "keyword_scrambled_model_data.h",
)

generate_cc_arrays(
    name = "generated_keyword_scrambled_model_cc",
    src = "keyword_scrambled.tflite",
    out = "keyword_scrambled_model_data.cc",
)

generate_cc_arrays(
    name = "generated_keyword_scrambled_8bit_model_hdr",
    src = "keyword_scrambled_8bit.tflite",
    out = "keyword_scrambled_8bit_model_data.h",
)

generate_cc_arrays(
    name = "generated_keyword_scrambled_8bit_model_cc",
    src = "keyword_scrambled_8bit.tflite",
    out = "keyword_scrambled_8bit_model_data.cc",
)
