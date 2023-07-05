load("//python/tflite_micro/signal:tflm_signal.bzl", "tflm_signal_kernel_library")

package(
    default_visibility = ["//python/tflite_micro/signal:__subpackages__"],
    licenses = ["notice"],
)

tflm_signal_kernel_library(
    name = "fft_kernel",
    srcs = ["fft_kernels.cc"],
    deps = [
        "//signal/src:rfft",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "framer_kernel",
    srcs = ["framer_kernel.cc"],
    deps = [
        "//signal/src:circular_buffer",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "overlap_add_kernel",
    srcs = ["overlap_add_kernel.cc"],
    deps = [
        "//signal/src:overlap_add",
        "@tensorflow_cc_deps//:cc_library",
    ],
)

tflm_signal_kernel_library(
    name = "window_kernel",
    srcs = ["window_kernel.cc"],
    deps = [
        "//signal/src:window",
        "@tensorflow_cc_deps//:cc_library",
    ],
)
