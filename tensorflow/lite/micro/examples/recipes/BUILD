load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    licenses = ["notice"],
)

py_binary(
    name = "variable",
    srcs = ["variable.py"],
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
    ],
)
