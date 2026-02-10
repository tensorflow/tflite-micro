def extra_tensorflow_targets():
    native.cc_library(
        name = "cc_headers",
        hdrs = native.glob(
            ["site-packages/tensorflow/include/**"],
            allow_empty = True,
        ),
        includes = ["site-packages/tensorflow/include"],
        visibility = ["//visibility:public"],
    )

    native.cc_library(
        name = "cc_library",
        srcs = ["site-packages/tensorflow/libtensorflow_framework.so.2"],
        deps = [":cc_headers"],
        visibility = ["//visibility:public"],
    )
