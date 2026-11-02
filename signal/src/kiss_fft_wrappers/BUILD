load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//signal:__subpackages__",
        "//tensorflow/lite/micro:__subpackages__",
    ],
    features = ["parse_headers"],
    licenses = ["notice"],
)

cc_library(
    name = "kiss_fft_wrappers",
    srcs = [
        "kiss_fft_float.cc",
        "kiss_fft_int16.cc",
        "kiss_fft_int32.cc",
    ],
    hdrs = [
        "kiss_fft_common.h",
        "kiss_fft_float.h",
        "kiss_fft_int16.h",
        "kiss_fft_int32.h",
    ],
    deps = [
        "//third_party/kissfft",
        "//third_party/kissfft:kissfftr",
    ],
)
