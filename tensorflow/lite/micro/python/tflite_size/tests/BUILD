load("@rules_python//python:defs.bzl", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

licenses(["notice"])

filegroup(
    name = "test_resources",
    srcs = [
        "gold_simple_add_model_html.txt",
        "gold_simple_add_model_json.txt",
        "simple_add_model.tflite",
    ],
)

py_test(
    name = "flatbuffer_size_test",
    srcs = ["flatbuffer_size_test.py"],
    data = [
        ":test_resources",
    ],
    main = "flatbuffer_size_test.py",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols from flatbuffer_size_wrapper_pybind.so
        "noubsan",
    ],
    deps = [
        requirement("tensorflow"),
        "//tensorflow/lite/micro/python/tflite_size/src:flatbuffer_size",
    ],
)
