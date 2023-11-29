package(
    default_visibility = ["//signal:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "complex",
    hdrs = ["complex.h"],
)

cc_library(
    name = "fft_auto_scale",
    srcs = ["fft_auto_scale.cc"],
    hdrs = ["fft_auto_scale.h"],
    deps = [
        ":max_abs",
        ":msb_32",
    ],
)

cc_library(
    name = "irfft",
    srcs = [
        "irfft_float.cc",
        "irfft_int16.cc",
        "irfft_int32.cc",
    ],
    hdrs = ["irfft.h"],
    deps = [
        ":complex",
        "//signal/src/kiss_fft_wrappers",
    ],
)

cc_library(
    name = "max_abs",
    srcs = ["max_abs.cc"],
    hdrs = ["max_abs.h"],
)

cc_library(
    name = "square_root_32",
    srcs = ["square_root_32.cc"],
    hdrs = ["square_root.h"],
    deps = [":msb_32"],
)

cc_library(
    name = "square_root_64",
    srcs = ["square_root_64.cc"],
    hdrs = ["square_root.h"],
    deps = [
        ":msb_64",
        ":square_root_32",
    ],
)

cc_library(
    name = "log",
    srcs = ["log.cc"],
    hdrs = ["log.h"],
    deps = [":msb_32"],
)

cc_library(
    name = "msb_32",
    srcs = ["msb_32.cc"],
    hdrs = ["msb.h"],
)

cc_library(
    name = "msb_64",
    srcs = ["msb_64.cc"],
    hdrs = ["msb.h"],
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

cc_library(
    name = "circular_buffer",
    srcs = ["circular_buffer.cc"],
    hdrs = ["circular_buffer.h"],
)

cc_library(
    name = "overlap_add",
    srcs = ["overlap_add.cc"],
    hdrs = ["overlap_add.h"],
)

cc_library(
    name = "energy",
    srcs = ["energy.cc"],
    hdrs = ["energy.h"],
    deps = [":complex"],
)

cc_library(
    name = "filter_bank",
    srcs = ["filter_bank.cc"],
    hdrs = ["filter_bank.h"],
)

cc_library(
    name = "filter_bank_log",
    srcs = ["filter_bank_log.cc"],
    hdrs = ["filter_bank_log.h"],
    deps = [
        ":log",
    ],
)

cc_library(
    name = "filter_bank_spectral_subtraction",
    srcs = ["filter_bank_spectral_subtraction.cc"],
    hdrs = ["filter_bank_spectral_subtraction.h"],
)

cc_library(
    name = "filter_bank_square_root",
    srcs = ["filter_bank_square_root.cc"],
    hdrs = ["filter_bank_square_root.h"],
    deps = [
        ":square_root_64",
    ],
)

cc_library(
    name = "pcan_argc_fixed",
    srcs = ["pcan_argc_fixed.cc"],
    hdrs = ["pcan_argc_fixed.h"],
    deps = [
        ":msb_32",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)
