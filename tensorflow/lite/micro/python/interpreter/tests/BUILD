load("@tflm_pip_deps//:requirements.bzl", "requirement")

licenses(["notice"])

py_library(
    name = "test_model_lib",
    srcs = ["test_model.py"],
    deps = [
        "//tensorflow/lite/python:schema_py",
        requirement("numpy"),
    ],
)

py_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    main = "interpreter_test.py",
    python_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols from interpreter_wrapper_pybind.so
        "notap",  # TODO(b/247808903)
        "noubsan",
    ],
    deps = [
        ":test_model_lib",
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
        "//tensorflow/lite/tools:flatbuffer_utils",
        requirement("tensorflow-cpu"),
    ],
)
