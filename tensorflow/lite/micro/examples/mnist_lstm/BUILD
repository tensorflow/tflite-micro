load("//python:py_rules.bzl", "tflm_py_binary", "tflm_py_test")
load("//tensorflow/lite/micro:build_def.bzl", "INCOMPATIBLE_WITH_WINDOWS")

package(default_applicable_licenses = ["//:license"])

tflm_py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        "//python/tflite_micro:runtime",
        "//third_party/py/PIL:pil",
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
        "//third_party/py/numpy",
        "//third_party/tensorflow/lite/python:lite",
    ],
)
