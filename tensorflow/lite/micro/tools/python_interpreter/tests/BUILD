load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "test_interpreter",
    main = "test_interpreter_bazel.py",
    srcs = ["test_interpreter_bazel.py"],
    python_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//tensorflow/lite/micro/tools/python_interpreter/src:tflm_runtime",
    ],
    data = [
        "//tensorflow/lite/micro/tools/python_interpreter/src:interpreter_wrapper_pybind.so",
    ],
)
