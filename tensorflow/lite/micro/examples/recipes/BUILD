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
    name = "resource_variables_test",
    srcs = ["resource_variables_test.py"],
    srcs_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols from interpreter_wrapper_pybind.so
        "noubsan",
    ],
    deps = [
        ":resource_variables_lib",
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
    ],
)
