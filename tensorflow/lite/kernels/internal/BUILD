load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/micro:build_def.bzl", "micro_copts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "common",
    srcs = [],
    hdrs = [
        "common.h",
        "optimized/neon_check.h",
            ],
    copts = tflite_copts(),
    deps = [
        ":cppmath",
        ":types",
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

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    copts = tflite_copts() + micro_copts(),
    deps = [
        ":compatibility",
        ":cppmath",
        ":types",
    ],
)

cc_library(
    name = "reference",
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
    name = "reference_base",
    srcs = [],
    hdrs = [
        "reference/add.h",
        "reference/add_n.h",
        "reference/arg_min_max.h",
        "reference/batch_to_space_nd.h",
        "reference/binary_function.h",
        "reference/ceil.h",
        "reference/comparisons.h",
        "reference/concatenation.h",
        "reference/conv.h",
        "reference/cumsum.h",
        "reference/depthwiseconv_float.h",
        "reference/depthwiseconv_uint8.h",
        "reference/dequantize.h",
        "reference/elu.h",
        "reference/exp.h",
        "reference/fill.h",
        "reference/floor.h",
        "reference/floor_div.h",
        "reference/floor_mod.h",
        "reference/fully_connected.h",
        "reference/hard_swish.h",
        "reference/integer_ops/add.h",
        "reference/integer_ops/conv.h",
        "reference/integer_ops/depthwise_conv.h",
        "reference/integer_ops/fully_connected.h",
        "reference/integer_ops/l2normalization.h",
        "reference/integer_ops/logistic.h",
        "reference/integer_ops/mean.h",
        "reference/integer_ops/mul.h",
        "reference/integer_ops/pooling.h",
        "reference/integer_ops/tanh.h",
        "reference/integer_ops/transpose_conv.h",
        "reference/l2normalization.h",
        "reference/leaky_relu.h",
        "reference/logistic.h",
        "reference/maximum_minimum.h",
        "reference/mul.h",
        "reference/neg.h",
        "reference/pad.h",
        "reference/pooling.h",
        "reference/prelu.h",
        "reference/process_broadcast_shapes.h",
        "reference/quantize.h",
        "reference/reduce.h",
        "reference/requantize.h",
        "reference/resize_nearest_neighbor.h",
        "reference/round.h",
        "reference/softmax.h",
        "reference/space_to_batch_nd.h",
        "reference/strided_slice.h",
        "reference/sub.h",
        "reference/tanh.h",
        "reference/transpose_conv.h",
    ],
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
    hdrs = ["types.h"],
    copts = tflite_copts(),
    deps = [
        ":compatibility",
    ],
)
