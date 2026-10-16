load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "tf_c_headers",
    hdrs = glob(["tensorflow/**/*.h"]),
    defines = ["IS_SLIM_BUILD=1"],
    includes = ["."],
)

# Targets matching //third_party/tensorflow/c in Google3 for 1-to-1 Copybara mapping.
cc_library(
    name = "ops",
    hdrs = ["tensorflow/c/ops.h"],
    deps = [":tf_c_headers"],
)

cc_library(
    name = "kernels",
    hdrs = ["tensorflow/c/kernels.h"],
    deps = [":tf_c_headers"],
)

cc_library(
    name = "tf_datatype",
    hdrs = ["tensorflow/c/tf_datatype.h"],
    deps = [":tf_c_headers"],
)

cc_library(
    name = "tf_status",
    hdrs = ["tensorflow/c/tf_status.h"],
    deps = [":tf_c_headers"],
)

cc_library(
    name = "tf_tensor",
    hdrs = ["tensorflow/c/tf_tensor.h"],
    deps = [":tf_c_headers"],
)
