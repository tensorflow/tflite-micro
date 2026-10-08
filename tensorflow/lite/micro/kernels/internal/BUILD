load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
)

package(
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
        "optimized/neon_check.h",
    ],
    deps = [
        ":compatibility",
        ":cppmath",
        ":types",
    ],
)

tflm_cc_library(
    name = "compatibility",
    hdrs = ["compatibility.h"],
    deps = [
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro/kernels:op_macros",
    ],
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
        ":compatibility",
        ":cppmath",
        ":types",
    ],
)

tflm_cc_library(
    name = "reference",
    srcs = ["tensor_ctypes.cc"],
    hdrs = [
        "portable_tensor.h",
        "tensor_ctypes.h",
    ],
    deps = [
        ":compatibility",
        ":types",
        "//tensorflow/lite/micro:micro_common",
    ],
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
    features = ["-parse_headers"],
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        ":quantization_util",
        ":strided_slice_logic",
        ":tensor",
        ":types",
        "//tensorflow/lite/micro:micro_common",
        "//tensorflow/lite/micro/kernels:kernel_util",
        "//tensorflow/lite/micro/kernels:op_macros",
    ],
)

tflm_cc_library(
    name = "strided_slice_logic",
    srcs = [],
    hdrs = [
        "strided_slice_logic.h",
    ],
    deps = [
        ":compatibility",
        ":types",
    ],
)

tflm_cc_library(
    name = "tensor",
    hdrs = [
        "portable_tensor.h",
        "tensor_ctypes.h",
    ],
    deps = [
        ":compatibility",
        ":types",
        "//tensorflow/lite/micro:micro_common",
    ],
)

tflm_cc_library(
    name = "types",
    hdrs = [
        "runtime_shape.h",
        "types.h",
    ],
    deps = [
        ":compatibility",
    ],
)

tflm_cc_library(
    name = "tensor_utils_no_eigen",
    srcs = [
        "portable_tensor_utils.cc",
    ],
    hdrs = [
        "portable_tensor_utils.h",
        "reference/portable_tensor_utils.h",
    ],
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        "//tensorflow/lite/micro:micro_common",
        "//tensorflow/lite/micro/kernels:op_macros",
    ],
)
