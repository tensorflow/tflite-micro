# Test data for signal tests. This is used to store large arrays which would make tests less readable.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//signal/micro/kernels:__subpackages__"],
    features = ["parse_headers"],
    licenses = ["notice"],
)

cc_library(
    name = "fft_test_data",
    srcs = [
        "fft_test_data.cc",
    ],
    hdrs = [
        "fft_test_data.h",
    ],
)
