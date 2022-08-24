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