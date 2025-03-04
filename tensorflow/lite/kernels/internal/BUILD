load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_copts",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = [
        "common.h",
        "optimized/neon_check.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":cppmath",
        ":types",
        "//tensorflow/lite/core:macros",
        "@gemmlowp//:fixedpoint",
    ],
)

cc_library(
    name = "compatibility",
    hdrs = ["compatibility.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/kernels:op_macros",
    ],
)

cc_library(
    name = "cppmath",
    srcs = [],
    hdrs = [
        "cppmath.h",
        "max.h",
        "min.h",
    ],
    copts = tflite_copts(),
)

tflm_cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    copts = tflm_copts() + tflite_copts(),
    deps = [
        ":compatibility",
        ":cppmath",
        ":types",
    ],
)

cc_library(
    name = "reference",
    srcs = ["tensor_ctypes.cc"],
    hdrs = [
        "portable_tensor.h",
        "tensor_ctypes.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:macros",
    ],
)

cc_library(
    name = "reference_base",
    srcs = glob([
        "reference/*.cc",
    ]),
    hdrs = glob([
        "reference/*.h",
        "reference/integer_ops/*.h",
    ]),
    copts = tflite_copts(),
    # We are disabling parse_headers for this header-only target so that the
    # external and internal builds are consistent. The primary issue here is
    # that parse_headers is not supported with bazel and the TFLM team would
    # really like to have all build errors be reproducible from the OSS build as
    # well.
    #
    # See b/175817116 for more details.
    features = ["-parse_headers"],
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        ":quantization_util",
        ":strided_slice_logic",
        ":tensor",
        ":types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:macros",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "@gemmlowp//:fixedpoint",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "strided_slice_logic",
    srcs = [],
    hdrs = [
        "strided_slice_logic.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":compatibility",
        ":types",
    ],
)

cc_library(
    name = "tensor",
    hdrs = [
        "portable_tensor.h",
        "tensor_ctypes.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":types",
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "types",
    hdrs = [
        "runtime_shape.h",
        "types.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":compatibility",
    ],
)

cc_library(
    name = "tensor_utils_no_eigen",
    srcs = [
        "portable_tensor_utils.cc",
        "reference/portable_tensor_utils.cc",
        "tensor_utils.cc",
    ],
    hdrs = [
        "portable_tensor_utils.h",
        "reference/portable_tensor_utils.h",
        "reference/portable_tensor_utils_impl.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:macros",
        "//tensorflow/lite/kernels:op_macros",
        "@gemmlowp",
    ],
)
