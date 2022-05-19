load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    data = [
        "//tensorflow/lite/micro/tools/python_interpreter/src:interpreter_wrapper_pybind.so",
    ],
    main = "interpreter_test.py",
    python_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//tensorflow/lite/micro/testing:generate_test_models",
        "//tensorflow/lite/micro/tools/python_interpreter/src:tflm_runtime",
    ],
)
