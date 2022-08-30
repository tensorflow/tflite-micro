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
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        requirement("pillow"),
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
    ],
)

py_test(
    name = "evaluate_test",
    srcs = ["evaluate_test.py"],
    data = glob(["samples/*.png"]),
    main = "evaluate_test.py",
    python_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        ":evaluate",
        ":train",
    ],
)
