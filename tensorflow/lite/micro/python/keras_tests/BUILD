# Description:
#   TensorFlow Lite microcontroller example.
load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

py_test(
    name = "conv_tests",
    srcs = ["conv_tests.py"],
    main = "conv_tests.py",
    python_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//python/tflite_micro:runtime",
    ],
)
