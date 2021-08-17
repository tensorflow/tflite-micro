# Description:
#   TensorFlow Lite for Microcontrollers Vision Example.
package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

load("//tensorflow/lite/micro:build_def.bzl","generate_cc_arrays")

generate_cc_arrays(
    name="generated_person_detect_model_cc",
    src="person_detect.tflite",
    out="person_detect_model_data.cc",
)

generate_cc_arrays(
    name="generated_person_detect_model_hdr",
    src="person_detect.tflite",
    out="person_detect_model_data.h",
)
