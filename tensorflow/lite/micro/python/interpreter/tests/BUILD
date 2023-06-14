load("@tflm_pip_deps//:requirements.bzl", "requirement")

licenses(["notice"])

py_test(
    name = "runtime_test",
    srcs = ["runtime_test.py"],
    python_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//tensorflow/lite/micro/python/interpreter/src:runtime",
        "//tensorflow/lite/micro/testing:generate_test_models_lib",
    ],
)
