# Description:
#   TensorFlow Lite for Microcontrollers Vision Example.
load("//tensorflow/lite/micro:build_def.bzl","generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

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
