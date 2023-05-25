# Description:
#   TensorFlow Lite for Microcontrollers Vision Example.
load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

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
