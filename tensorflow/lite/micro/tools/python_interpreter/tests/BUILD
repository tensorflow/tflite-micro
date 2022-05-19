load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "interpreter_test",
    main = "interpreter_test.py",
    srcs = ["interpreter_test.py"],
    python_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//tensorflow/lite/micro/tools/python_interpreter/src:tflm_runtime",
        "//tensorflow/lite/micro/testing:generate_test_models",
    ],
    data = [
        "//tensorflow/lite/micro/tools/python_interpreter/src:interpreter_wrapper_pybind.so",
    ],
)
