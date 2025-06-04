# Description:
#   TensorFlow Lite microcontroller example.
load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "tflm_cc_library",
    "tflm_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

generate_cc_arrays(
    name = "generated_yes_1000ms_wav_cc",
    src = "testdata/yes_1000ms.wav",
    out = "testdata/yes_1000ms_audio_data.cc",
)

generate_cc_arrays(
    name = "generated_yes_1000ms_wav_hdr",
    src = "testdata/yes_1000ms.wav",
    out = "testdata/yes_1000ms_audio_data.h",
)

generate_cc_arrays(
    name = "generated_no_1000ms_wav_cc",
    src = "testdata/no_1000ms.wav",
    out = "testdata/no_1000ms_audio_data.cc",
)

generate_cc_arrays(
    name = "generated_no_1000ms_wav_hdr",
    src = "testdata/no_1000ms.wav",
    out = "testdata/no_1000ms_audio_data.h",
)

generate_cc_arrays(
    name = "generated_noise_1000ms_wav_cc",
    src = "testdata/noise_1000ms.wav",
    out = "testdata/noise_1000ms_audio_data.cc",
)

generate_cc_arrays(
    name = "generated_noise_1000ms_wav_hdr",
    src = "testdata/noise_1000ms.wav",
    out = "testdata/noise_1000ms_audio_data.h",
)

generate_cc_arrays(
    name = "generated_silence_1000ms_wav_cc",
    src = "testdata/silence_1000ms.wav",
    out = "testdata/silence_1000ms_audio_data.cc",
)

generate_cc_arrays(
    name = "generated_silence_1000ms_wav_hdr",
    src = "testdata/silence_1000ms.wav",
    out = "testdata/silence_1000ms_audio_data.h",
)

generate_cc_arrays(
    name = "generated_yes_30ms_wav_cc",
    src = "testdata/yes_30ms.wav",
    out = "testdata/yes_30ms_audio_data.cc",
)

generate_cc_arrays(
    name = "generated_yes_30ms_wav_hdr",
    src = "testdata/yes_30ms.wav",
    out = "testdata/yes_30ms_audio_data.h",
)

generate_cc_arrays(
    name = "generated_no_30ms_wav_cc",
    src = "testdata/no_30ms.wav",
    out = "testdata/no_30ms_audio_data.cc",
)

generate_cc_arrays(
    name = "generated_no_30ms_wav_hdr",
    src = "testdata/no_30ms.wav",
    out = "testdata/no_30ms_audio_data.h",
)

generate_cc_arrays(
    name = "generated_micro_speech_model_cc",
    src = "models/micro_speech_quantized.tflite",
    out = "models/micro_speech_quantized_model_data.cc",
)

generate_cc_arrays(
    name = "generated_micro_speech_model_hdr",
    src = "models/micro_speech_quantized.tflite",
    out = "models/micro_speech_quantized_model_data.h",
)

generate_cc_arrays(
    name = "generated_audio_preprocessor_model_cc",
    src = "models/audio_preprocessor_int8.tflite",
    out = "models/audio_preprocessor_int8_model_data.cc",
)

generate_cc_arrays(
    name = "generated_audio_preprocessor_model_hdr",
    src = "models/audio_preprocessor_int8.tflite",
    out = "models/audio_preprocessor_int8_model_data.h",
)

tflm_cc_library(
    name = "micro_speech_model_data",
    srcs = [
        ":generated_micro_speech_model_cc",
    ],
    hdrs = [
        ":generated_micro_speech_model_hdr",
    ],
)

tflm_cc_library(
    name = "audio_preprocessor_model_data",
    srcs = [
        ":generated_audio_preprocessor_model_cc",
    ],
    hdrs = [
        ":generated_audio_preprocessor_model_hdr",
    ],
)

tflm_cc_library(
    name = "audio_sample_test_data_30ms",
    srcs = [
        ":generated_no_30ms_wav_cc",
        ":generated_yes_30ms_wav_cc",
    ],
    hdrs = [
        ":generated_no_30ms_wav_hdr",
        ":generated_yes_30ms_wav_hdr",
    ],
)

tflm_cc_library(
    name = "audio_sample_test_data_1000ms",
    srcs = [
        ":generated_no_1000ms_wav_cc",
        ":generated_noise_1000ms_wav_cc",
        ":generated_silence_1000ms_wav_cc",
        ":generated_yes_1000ms_wav_cc",
    ],
    hdrs = [
        ":generated_no_1000ms_wav_hdr",
        ":generated_noise_1000ms_wav_hdr",
        ":generated_silence_1000ms_wav_hdr",
        ":generated_yes_1000ms_wav_hdr",
    ],
)

tflm_cc_library(
    name = "micro_model_settings",
    hdrs = [
        "micro_model_settings.h",
    ],
)

tflm_cc_test(
    name = "micro_speech_test",
    srcs = [
        "micro_speech_test.cc",
    ],
    deps = [
        ":audio_preprocessor_model_data",
        ":audio_sample_test_data_1000ms",
        ":audio_sample_test_data_30ms",
        ":micro_model_settings",
        ":micro_speech_model_data",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

filegroup(
    name = "samples_30ms",
    srcs = glob(["testdata/*_30ms.wav"]),
)

filegroup(
    name = "samples_1000ms",
    srcs = glob(["testdata/*_1000ms.wav"]),
)

filegroup(
    name = "models_tflite",
    srcs = glob(["models/*.tflite"]),
)

py_binary(
    name = "audio_preprocessor",
    srcs = ["audio_preprocessor.py"],
    data = [
        ":samples_30ms",
    ],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        requirement("numpy"),
        requirement("tensorflow"),
        "//python/tflite_micro:runtime",
        "//python/tflite_micro/signal:ops",
        "//python/tflite_micro/signal/utils:util",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    data = [
        ":models_tflite",
    ],
    deps = [
        ":audio_preprocessor",
    ],
)

py_test(
    name = "evaluate_test",
    srcs = ["evaluate_test.py"],
    data = [
        ":models_tflite",
        ":samples_1000ms",
    ],
    main = "evaluate_test.py",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    deps = [
        ":evaluate",
    ],
)

py_test(
    name = "audio_preprocessor_test",
    srcs = ["audio_preprocessor_test.py"],
    data = [
        ":models_tflite",
        ":samples_30ms",
    ],
    main = "audio_preprocessor_test.py",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    deps = [
        ":audio_preprocessor",
    ],
)
