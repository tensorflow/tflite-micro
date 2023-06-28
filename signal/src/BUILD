package(
    default_visibility = ["//signal:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "complex",
    hdrs = ["complex.h"],
)

cc_library(
    name = "rfft",
    srcs = [
        "rfft_float.cc",
        "rfft_int16.cc",
        "rfft_int32.cc",
    ],
    hdrs = ["rfft.h"],
    deps = [
        ":complex",
        "//signal/src/kiss_fft_wrappers",
    ],
)

cc_library(
    name = "window",
    srcs = ["window.cc"],
    hdrs = ["window.h"],
)
