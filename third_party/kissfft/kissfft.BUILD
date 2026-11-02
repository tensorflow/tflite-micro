load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-parse_headers",
        "-use_header_modules",
    ],
)

licenses(["notice"])  # BSD-3-Clause

exports_files([
    "COPYING",
    "LICENSES/BSD-3-Clause",
])

cc_library(
    name = "kiss_fft",
    textual_hdrs = [
        "_kiss_fft_guts.h",
        "kiss_fft.c",
        "kiss_fft.h",
        "kiss_fft_log.h",
    ],
)

cc_library(
    name = "kiss_fftr",
    textual_hdrs = [
        "kiss_fftr.c",
        "kiss_fftr.h",
    ],
    deps = [":kiss_fft"],
)
