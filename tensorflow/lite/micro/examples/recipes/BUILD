load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    licenses = ["notice"],
)

py_library(
    name = "resource_variables_lib",
    srcs = ["resource_variables_lib.py"],
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
    ],
)

py_test(
    name = "recipes_test",
    srcs = ["recipes_test.py"],
    srcs_version = "PY3",
    deps = [
        ":resource_variables_lib",
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
    ],
)
