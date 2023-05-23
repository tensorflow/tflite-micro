load("//signal:tflm_signal.bzl", "tflm_signal_kernel_library")

package(
    default_visibility = ["//signal:__subpackages__"],
    licenses = ["notice"],
)

tflm_signal_kernel_library(
    name = "window_kernel",
    srcs = ["window_kernel.cc"],
    deps = [
        "//signal/src:window",
        "@tensorflow_headers",
        "@tensorflow_headers//:tensorflow_headers_lib"
    ],
)