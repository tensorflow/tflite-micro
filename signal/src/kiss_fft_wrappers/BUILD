package(
    default_visibility = ["//signal/src:__subpackages__"],
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
        "@kissfft//:kiss_fftr",
    ],
)
