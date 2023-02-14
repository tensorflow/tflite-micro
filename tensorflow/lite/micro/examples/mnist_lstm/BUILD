load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
        "@absl_py//absl:app",
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
    python_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols from interpreter_wrapper_pybind.so
        "noubsan",
    ],
    deps = [
        ":evaluate",
        ":train",
        "//tensorflow/lite/micro/tools:requantize_flatbuffer",
    ],
)
