load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = [
        "common.h",
        "fixedpoint.h",
    ],
    deps = [
        ":cppmath",
        ":types",
        "//tensorflow/lite/micro:micro_compatibility",
    ],
)

tflm_cc_library(
    name = "compatibility",
    hdrs = ["compatibility.h"],
    deps = ["//tensorflow/lite/micro:micro_compatibility"],
)

tflm_cc_library(
    name = "cppmath",
    srcs = [],
    hdrs = [
        "cppmath.h",
        "max.h",
        "min.h",
    ],
)

tflm_cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    deps = [
        ":cppmath",
        ":types",
        "//tensorflow/lite/micro:micro_compatibility",
    ],
)

alias(
    name = "reference",
    actual = ":tensor",
)

tflm_cc_library(
    name = "reference_base",
    srcs = glob(
        [
            "reference/*.cc",
        ],
        allow_empty = True,
    ),
    hdrs = glob([
        "reference/*.h",
        "reference/integer_ops/*.h",
    ]),
    deps = [
        ":common",
        ":cppmath",
        ":quantization_util",
        ":strided_slice_logic",
        ":tensor",
        ":tensor_utils_no_eigen",
        ":types",
        "//tensorflow/lite/micro:micro_common",
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro/kernels:kernel_util",
    ],
)

tflm_cc_library(
    name = "strided_slice_logic",
    srcs = [],
    hdrs = [
        "strided_slice_logic.h",
    ],
    deps = [
        ":types",
        "//tensorflow/lite/micro:micro_compatibility",
    ],
)

tflm_cc_library(
    name = "tensor",
    srcs = ["tensor_ctypes.cc"],
    hdrs = [
        "portable_tensor.h",
        "tensor_ctypes.h",
    ],
    deps = [
        ":types",
        "//tensorflow/lite/micro:micro_common",
        "//tensorflow/lite/micro:micro_compatibility",
    ],
)

tflm_cc_library(
    name = "types",
    hdrs = [
        "runtime_shape.h",
        "types.h",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_compatibility",
    ],
)

tflm_cc_library(
    name = "tensor_utils_no_eigen",
    srcs = [
        "portable_tensor_utils.cc",
    ],
    hdrs = [
        "portable_tensor_utils.h",
    ],
    deps = [
        ":common",
        ":cppmath",
        "//tensorflow/lite/micro:micro_common",
        "//tensorflow/lite/micro:micro_compatibility",
    ],
)
