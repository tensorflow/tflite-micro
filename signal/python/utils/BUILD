# Signal python utilities.
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = [
        "//signal:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
        requirement("tensorflow-cpu"),
    ],
)
