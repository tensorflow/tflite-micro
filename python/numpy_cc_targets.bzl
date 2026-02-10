def extra_numpy_targets():
    native.cc_library(
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
