# Description:
#   TensorFlow Lite microcontrollers lstm example.

load("//tensorflow/lite/micro:build_def.bzl","generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

generate_cc_arrays(
    name = "generated_micro_speech_lstm_model_cc",
    src = "micro_speech_lstm.tflite",
    out = "micro_speech_lstm_model_data.cc",
)

generate_cc_arrays(
    name = "generated_micro_speech_lstm_model_hdr",
    src = "micro_speech_lstm.tflite",
    out = "micro_speech_lstm_model_data.h",
)
