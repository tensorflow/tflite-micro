load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
)
load(
    "//tensorflow:extra_rules.bzl",
    "tflm_python_op_resolver_friends",
)

package(
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "op_resolver_friends",
    packages = tflm_python_op_resolver_friends(),
)

# TODO(b/286456378): remove once all internal usage is fixed.
tflm_cc_library(
    name = "python_ops_resolver",
    srcs = [],
    hdrs = [
        "python_ops_resolver.h",
    ],
    visibility = [":op_resolver_friends"],
    deps = [
        "//python/tflite_micro:python_ops_resolver",
    ],
)
