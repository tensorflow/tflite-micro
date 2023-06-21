load("//python/tflite_micro/signal:tflm_signal.bzl", "tflm_signal_kernel_library")

package(
    default_visibility = ["//python/tflite_micro/signal:__subpackages__"],
    licenses = ["notice"],
)

tflm_signal_kernel_library(
    name = "fft_ops",
    srcs = ["fft_ops.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "window_op",
    srcs = ["window_op.cc"],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)
