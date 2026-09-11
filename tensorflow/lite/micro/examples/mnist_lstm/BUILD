load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_rules.bzl", "tflm_py_binary", "tflm_py_test")
load("//tensorflow/lite/micro:build_def.bzl", "INCOMPATIBLE_WITH_WINDOWS")

tflm_py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

tflm_py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        "//python/tflite_micro:runtime",
        requirement("pillow"),
    ],
)

filegroup(
    name = "sample_images",
    srcs = glob(["samples/*.png"]),
)

tflm_py_test(
    name = "evaluate_test",
    srcs = ["evaluate_test.py"],
    data = [
        "trained_lstm.tflite",
        "trained_lstm_int8.tflite",
        ":sample_images",
    ],
    main = "evaluate_test.py",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":evaluate",
        "//tensorflow/lite/micro/tools:requantize_flatbuffer",
        requirement("ai-edge-litert"),
        requirement("numpy"),
    ],
)
