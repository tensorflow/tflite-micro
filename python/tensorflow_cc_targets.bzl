load("@rules_cc//cc:cc_library.bzl", "cc_library")

def extra_tensorflow_targets():
    cc_library(
        name = "cc_headers",
        hdrs = native.glob(
            ["site-packages/tensorflow/include/**"],
            allow_empty = True,
        ),
        includes = ["site-packages/tensorflow/include"],
        visibility = ["//visibility:public"],
    )

    cc_library(
        name = "cc_library",
        srcs = ["site-packages/tensorflow/libtensorflow_framework.so.2"],
        deps = [":cc_headers"],
        visibility = ["//visibility:public"],
    )
