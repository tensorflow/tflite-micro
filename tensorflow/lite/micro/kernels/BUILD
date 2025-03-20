load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_cc_test",
    "tflm_copts",
    "tflm_kernel_cc_library",
)
load(
    "//tensorflow:extra_rules.bzl",
    "tflm_kernel_friends",
    "xtensa_fusion_f1_config",
    "xtensa_hifi_3_config",
    "xtensa_hifi_3z_config",
    "xtensa_hifi_5_config",
    "xtensa_vision_p6_config",
)

package(
    features = [
        "-layering_check",  # buildozer: disable=no-layering-check, TODO(b/177257333): consider enabling layering check
        "-parse_headers",  # buildozer: disable=no-parse-headers, paser_headers is unavailable with bazel (http://b/175817117#comment4)
    ],
    licenses = ["notice"],
)

package_group(
    name = "tflite_micro",
    packages = ["//..."],
)

package_group(
    name = "micro_top_level",
    packages = ["//tensorflow/lite/micro"],
)

package_group(
    name = "kernel_friends",
    packages = tflm_kernel_friends(),
)

####################################
# C++ libraries
####################################

tflm_cc_library(
    name = "activation_utils",
    hdrs = ["activation_utils.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:cppmath",
    ],
)

tflm_cc_library(
    name = "circular_buffer_flexbuffers_generated_data",
    srcs = [
        "circular_buffer_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "circular_buffer_flexbuffers_generated_data.h",
    ],
)

tflm_cc_library(
    name = "conv_test_common",
    srcs = [
        "conv_test_common.cc",
    ],
    hdrs = [
        "conv_test.h",
    ],
    deps = [
        ":kernel_runner",
        ":micro_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_library(
    name = "decompress",
    srcs = [
        "decompress.cc",
        "decompress_common.cc",
    ],
    hdrs = [
        "decompress.h",
    ],
    visibility = [
        ":kernel_friends",
        ":tflite_micro",
    ],
    deps = [
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro:compression",
        "//tensorflow/lite/micro:micro_common",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_profiler",
    ],
)

tflm_cc_library(
    name = "detection_postprocess_flexbuffers_generated_data",
    srcs = [
        "detection_postprocess_flexbuffers_generated_data.cc",
    ],
    hdrs = [
        "detection_postprocess_flexbuffers_generated_data.h",
    ],
)

tflm_cc_library(
    name = "kernel_runner",
    srcs = [
        "kernel_runner.cc",
    ],
    hdrs = ["kernel_runner.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro:fake_micro_context",
        "//tensorflow/lite/micro:micro_allocator",
        "//tensorflow/lite/micro:micro_arena_constants",
        "//tensorflow/lite/micro:mock_micro_graph",
        "//tensorflow/lite/micro:test_helpers",
    ],
)

tflm_cc_library(
    name = "kernel_util",
    srcs = [
        "kernel_util.cc",
    ],
    hdrs = ["kernel_util.h"],
    visibility = [
        ":kernel_friends",
        ":tflite_micro",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor_utils_no_eigen",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:micro_context",
    ],
)

tflm_cc_library(
    name = "lstm_shared",
    hdrs = [
        "lstm_shared.h",
    ],
    visibility = ["//tensorflow/lite/micro/kernels/testdata:__pkg__"],
)

tflm_cc_library(
    name = "lstm_eval_test_lib",
    hdrs = [
        "lstm_eval_test.h",
    ],
    deps = [
        ":kernel_util",
        ":micro_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels/testdata:lstm_test_data",
    ],
)

tflm_cc_library(
    name = "micro_tensor_utils",
    srcs = [
        "micro_tensor_utils.cc",
    ],
    hdrs = ["micro_tensor_utils.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:macros",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite/kernels/internal:tensor_utils_no_eigen",
    ],
)

HIFI3_COPTS = [
    "-DXTENSA=1",
    "-DHIFI3=1",
]

HIFI4_COPTS = [
    "-DXTENSA=1",
    "-DHIFI4=1",
]

HIFI5_COPTS = [
    "-DXTENSA=1",
    "-DHIFI5=1",
]

VP6_COPTS = [
    "-DXTENSA=1",
    "-DVISION_P6=1",
]

tflm_kernel_cc_library(
    name = "micro_ops",
    srcs = [
        "activations.cc",
        "activations_common.cc",
        "add.cc",
        "add_common.cc",
        "add_n.cc",
        "arg_min_max.cc",
        "assign_variable.cc",
        "batch_matmul.cc",
        "batch_matmul_common.cc",
        "batch_to_space_nd.cc",
        "broadcast_args.cc",
        "broadcast_to.cc",
        "call_once.cc",
        "cast.cc",
        "ceil.cc",
        "circular_buffer.cc",
        "circular_buffer_common.cc",
        "comparisons.cc",
        "concatenation.cc",
        "conv.cc",
        "conv_common.cc",
        "cumsum.cc",
        "depth_to_space.cc",
        "depthwise_conv.cc",
        "depthwise_conv_common.cc",
        "dequantize.cc",
        "dequantize_common.cc",
        "detection_postprocess.cc",
        "div.cc",
        "elementwise.cc",
        "elu.cc",
        "embedding_lookup.cc",
        "ethosu.cc",
        "exp.cc",
        "expand_dims.cc",
        "fill.cc",
        "floor.cc",
        "floor_div.cc",
        "floor_mod.cc",
        "fully_connected.cc",
        "fully_connected_common.cc",
        "gather.cc",
        "gather_nd.cc",
        "hard_swish.cc",
        "hard_swish_common.cc",
        "if.cc",
        "l2_pool_2d.cc",
        "l2norm.cc",
        "leaky_relu.cc",
        "leaky_relu_common.cc",
        "log_softmax.cc",
        "logical.cc",
        "logical_common.cc",
        "logistic.cc",
        "logistic_common.cc",
        "lstm_eval.cc",
        "lstm_eval_common.cc",
        "maximum_minimum.cc",
        "mirror_pad.cc",
        "mul.cc",
        "mul_common.cc",
        "neg.cc",
        "pack.cc",
        "pad.cc",
        "pad_common.cc",
        "pooling.cc",
        "pooling_common.cc",
        "prelu.cc",
        "prelu_common.cc",
        "quantize.cc",
        "quantize_common.cc",
        "read_variable.cc",
        "reduce.cc",
        "reduce_common.cc",
        "reshape.cc",
        "reshape_common.cc",
        "resize_bilinear.cc",
        "resize_nearest_neighbor.cc",
        "round.cc",
        "select.cc",
        "shape.cc",
        "slice.cc",
        "softmax.cc",
        "softmax_common.cc",
        "space_to_batch_nd.cc",
        "space_to_depth.cc",
        "split.cc",
        "split_v.cc",
        "squared_difference.cc",
        "squeeze.cc",
        "strided_slice.cc",
        "strided_slice_common.cc",
        "sub.cc",
        "sub_common.cc",
        "svdf.cc",
        "svdf_common.cc",
        "tanh.cc",
        "transpose.cc",
        "transpose_common.cc",
        "transpose_conv.cc",
        "unidirectional_sequence_lstm.cc",
        "unpack.cc",
        "var_handle.cc",
        "while.cc",
        "zeros_like.cc",
    ],
    hdrs = [
        "activations.h",
        "add.h",
        "batch_matmul.h",
        "circular_buffer.h",
        "conv.h",
        "depthwise_conv.h",
        "dequantize.h",
        "ethosu.h",
        "fully_connected.h",
        "hard_swish.h",
        "leaky_relu.h",
        "logical.h",
        "logistic.h",
        "lstm_eval.h",
        "lstm_shared.h",
        "maximum_minimum.h",
        "micro_ops.h",
        "mul.h",
        "pad.h",
        "pooling.h",
        "prelu.h",
        "quantize.h",
        "reduce.h",
        "reshape.h",
        "softmax.h",
        "strided_slice.h",
        "sub.h",
        "svdf.h",
        "transpose.h",
        "transpose_conv.h",
        "unidirectional_sequence_lstm.h",
    ] + select({
        xtensa_fusion_f1_config(): glob(["xtensa/**/*.h"]),
        xtensa_hifi_3_config(): glob(["xtensa/**/*.h"]),
        xtensa_hifi_3z_config(): glob(["xtensa/**/*.h"]),
        xtensa_hifi_5_config(): glob(["xtensa/**/*.h"]),
        xtensa_vision_p6_config(): glob(["xtensa/**/*.h"]),
        "//conditions:default": [],
    }),
    accelerated_srcs = {
        xtensa_fusion_f1_config(): glob(["xtensa/**/*.cc"]),
        xtensa_hifi_3_config(): glob(["xtensa/**/*.cc"]),
        xtensa_hifi_3z_config(): glob(["xtensa/**/*.cc"]),
        xtensa_hifi_5_config(): glob(["xtensa/**/*.cc"]),
        xtensa_vision_p6_config(): glob(["xtensa/**/*.cc"]),
    },
    copts = tflm_copts() + select({
        xtensa_fusion_f1_config(): HIFI4_COPTS,
        xtensa_hifi_3_config(): HIFI3_COPTS,
        xtensa_hifi_3z_config(): HIFI4_COPTS,
        xtensa_hifi_5_config(): HIFI5_COPTS,
        xtensa_vision_p6_config(): VP6_COPTS,
        "//conditions:default": [],
    }),
    visibility = [
        # Public visibility to allow application code to select kernel variants.
        "//visibility:public",
    ],
    deps = [
        ":activation_utils",
        ":kernel_util",
        ":micro_tensor_utils",
        "//signal/micro/kernels:register_signal_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:tensor_utils_no_eigen",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/micro:flatbuffer_utils",
        "//tensorflow/lite/micro:memory_helpers",
        "//tensorflow/lite/micro:micro_graph",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ] + select({
        xtensa_fusion_f1_config(): ["//third_party/xtensa/nnlib_hifi4:nnlib_hifi4_lib"],
        xtensa_hifi_3_config(): ["//third_party/xtensa/nnlib_hifi4:nnlib_hifi4_lib"],
        xtensa_hifi_3z_config(): ["//third_party/xtensa/nnlib_hifi4:nnlib_hifi4_lib"],
        xtensa_hifi_5_config(): ["//third_party/xtensa/nnlib_hifi5:nnlib_hifi5_lib"],
        xtensa_vision_p6_config(): ["//third_party/xtensa/xi_tflmlib_vision_p6:xi_tflmlib_vision_p6_lib"],
        "//conditions:default": [],
    }),
)

####################################
# C++ tests
####################################

tflm_cc_test(
    name = "activations_test",
    srcs = [
        "activations_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "add_n_test",
    srcs = [
        "add_n_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "add_test",
    srcs = [
        "add_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "arg_min_max_test",
    srcs = [
        "arg_min_max_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "batch_matmul_test",
    srcs = [
        "batch_matmul_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "batch_to_space_nd_test",
    srcs = [
        "batch_to_space_nd_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "broadcast_args_test",
    srcs = [
        "broadcast_args_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "broadcast_to_test",
    srcs = [
        "broadcast_to_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "call_once_test",
    srcs = ["call_once_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "cast_test",
    srcs = ["cast_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "ceil_test",
    srcs = [
        "ceil_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "circular_buffer_test",
    srcs = [
        "circular_buffer_test.cc",
    ],
    deps = [
        "circular_buffer_flexbuffers_generated_data",
        ":kernel_runner",
        ":micro_ops",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "comparisons_test",
    srcs = [
        "comparisons_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "concatenation_test",
    srcs = [
        "concatenation_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "conv_test",
    srcs = [
        "conv_test.cc",
    ],
    deps = [
        ":conv_test_common",
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/kernels/testdata:conv_test_data",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "cumsum_test",
    srcs = [
        "cumsum_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "decompress_test",
    srcs = [
        "decompress_test.cc",
    ],
    tags = [
        "nobuilder",
        "notap",
    ],
    target_compatible_with = select({
        "//conditions:default": ["@platforms//:incompatible"],
        "//:with_compression_enabled": [],
    }),
    deps = [
        ":decompress",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_arena_constants",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "depth_to_space_test",
    srcs = [
        "depth_to_space_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "depthwise_conv_test",
    srcs = [
        "depthwise_conv_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "dequantize_test",
    srcs = [
        "dequantize_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "detection_postprocess_test",
    srcs = [
        "detection_postprocess_test.cc",
    ],
    deps = [
        ":detection_postprocess_flexbuffers_generated_data",
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
        "@flatbuffers//:runtime_cc",
    ],
)

tflm_cc_test(
    name = "div_test",
    srcs = [
        "div_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "elu_test",
    srcs = [
        "elu_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "embedding_lookup_test",
    srcs = [
        "embedding_lookup_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "exp_test",
    srcs = ["exp_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "expand_dims_test",
    srcs = ["expand_dims_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "fill_test",
    srcs = [
        "fill_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "floor_div_test",
    srcs = ["floor_div_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "floor_mod_test",
    srcs = ["floor_mod_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "floor_test",
    srcs = [
        "floor_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "fully_connected_test",
    srcs = [
        "fully_connected_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "gather_test",
    srcs = [
        "gather_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "gather_nd_test",
    srcs = [
        "gather_nd_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "hard_swish_test",
    srcs = ["hard_swish_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "if_test",
    srcs = ["if_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:mock_micro_graph",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "l2norm_test",
    srcs = [
        "l2norm_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "l2_pool_2d_test",
    srcs = [
        "l2_pool_2d_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "leaky_relu_test",
    srcs = [
        "leaky_relu_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "logical_test",
    srcs = [
        "logical_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "logistic_test",
    srcs = [
        "logistic_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "log_softmax_test",
    srcs = [
        "log_softmax_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "lstm_eval_test",
    srcs = [
        "lstm_eval_test.cc",
    ],
    deps = [
        ":lstm_eval_test_lib",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "maximum_minimum_test",
    srcs = [
        "maximum_minimum_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "mirror_pad_test",
    srcs = [
        "mirror_pad_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "mul_test",
    srcs = [
        "mul_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "neg_test",
    srcs = [
        "neg_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "pack_test",
    srcs = [
        "pack_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "pad_test",
    srcs = [
        "pad_test.cc",
    ],
    tags = [
        "noasan",
        "nomsan",  # TODO(b/175133159): currently failing with asan and msan
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "pooling_test",
    srcs = [
        "pooling_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "prelu_test",
    srcs = [
        "prelu_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "quantization_util_test",
    srcs = [
        "quantization_util_test.cc",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "quantize_test",
    srcs = [
        "quantize_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "reduce_test",
    srcs = [
        "reduce_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "reshape_test",
    srcs = [
        "reshape_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "resize_bilinear_test",
    srcs = [
        "resize_bilinear_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "resize_nearest_neighbor_test",
    srcs = [
        "resize_nearest_neighbor_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "round_test",
    srcs = [
        "round_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "select_test",
    srcs = [
        "select_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "shape_test",
    srcs = ["shape_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "slice_test",
    srcs = ["slice_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "softmax_test",
    srcs = [
        "softmax_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "space_to_batch_nd_test",
    srcs = [
        "space_to_batch_nd_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "space_to_depth_test",
    srcs = [
        "space_to_depth_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "split_test",
    srcs = [
        "split_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "split_v_test",
    srcs = [
        "split_v_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "squared_difference_test",
    srcs = [
        "squared_difference_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "squeeze_test",
    srcs = ["squeeze_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "strided_slice_test",
    srcs = [
        "strided_slice_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "sub_test",
    srcs = [
        "sub_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "svdf_test",
    srcs = [
        "svdf_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "transpose_conv_test",
    srcs = [
        "transpose_conv_test.cc",
    ],
    deps = [
        ":conv_test_common",
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "unidirectional_sequence_lstm_test",
    srcs = [
        "unidirectional_sequence_lstm_test.cc",
    ],
    deps = [
        ":kernel_runner",
        ":lstm_eval_test_lib",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "unpack_test",
    srcs = [
        "unpack_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "while_test",
    srcs = [
        "while_test.cc",
    ],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_test(
    name = "zeros_like_test",
    srcs = ["zeros_like_test.cc"],
    deps = [
        ":kernel_runner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:debug_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

####################################
# Bazel config settings.
####################################

# Command line flag to select which set of optimized kernels to use.
# Each value should have a `config_setting` which is selected on in the
# `micro_ops` target to pickup optimized kernel sources. An empty value
# indicates only reference kernels should be used.
string_flag(
    name = "optimized_kernels",
    build_setting_default = "",
    values = [
        "",
        "xtensa_fusion_f1",
        "xtensa_hifi_3",
        "xtensa_hifi_3z",
        "xtensa_hifi_5",
        "xtensa_vision_p6",
    ],
)

config_setting(
    name = "xtensa_fusion_f1_default",
    flag_values = {
        ":optimized_kernels": "xtensa_fusion_f1",
    },
)

config_setting(
    name = "xtensa_hifi_3_default",
    flag_values = {
        ":optimized_kernels": "xtensa_hifi_3",
    },
)

config_setting(
    name = "xtensa_hifi_3z_default",
    flag_values = {
        ":optimized_kernels": "xtensa_hifi_3z",
    },
)

config_setting(
    name = "xtensa_hifi_5_default",
    flag_values = {
        ":optimized_kernels": "xtensa_hifi_5",
    },
)

config_setting(
    name = "xtensa_vision_p6_default",
    flag_values = {
        ":optimized_kernels": "xtensa_vision_p6",
    },
)
