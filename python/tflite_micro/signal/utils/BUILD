# Signal python utilities.
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = [
        "//python/tflite_micro/signal:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//python/tflite_micro:runtime",
        "//python/tflite_micro/signal:ops",
        requirement("tensorflow-cpu"),
    ],
)
