load("@rules_cc//cc:cc_library.bzl", "cc_library")

def extra_numpy_targets():
    cc_library(
        name = "cc_headers",
        hdrs = native.glob(
            [
                "site-packages/numpy/_core/include/**",
                "site-packages/numpy/core/include/**",
            ],
            allow_empty = True,
        ),
        includes = [
            "site-packages/numpy/_core/include",
            "site-packages/numpy/core/include",
        ],
        visibility = ["//visibility:public"],
    )
