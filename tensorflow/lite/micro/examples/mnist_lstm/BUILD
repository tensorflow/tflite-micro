load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        "//python/tflite_micro:runtime",
        "@absl_py//absl:app",
        requirement("pillow"),
    ],
)

filegroup(
    name = "sample_images",
    srcs = glob(["samples/*.png"]),
)

py_test(
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
    deps = [
        ":evaluate",
        ":train",
        "//tensorflow/lite/micro/tools:requantize_flatbuffer",
    ],
)
