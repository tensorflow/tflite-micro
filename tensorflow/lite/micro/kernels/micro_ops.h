/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_MICRO_OPS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_MICRO_OPS_H_

#include "tensorflow/lite/c/common.h"

// Forward declaration of all micro op kernel registration methods. These
// registrations are included with the standard `BuiltinOpResolver`.
//
// This header is particularly useful in cases where only a subset of ops are
// needed. In such cases, the client can selectively add only the registrations
// their model requires, using a custom `(Micro)MutableOpResolver`. Selective
// registration in turn allows the linker to strip unused kernels.

namespace tflite {

// TFLM is incrementally moving towards a flat tflite namespace
// (https://abseil.io/tips/130). Any new ops (or cleanup of existing ops should
// have their Register function declarations in the tflite namespace.

TfLiteRegistration_V1 Register_ABS();
TfLiteRegistration_V1 Register_ADD();
TfLiteRegistration_V1 Register_ADD_N();
TfLiteRegistration_V1 Register_ARG_MAX();
TfLiteRegistration_V1 Register_ARG_MIN();
TfLiteRegistration_V1 Register_ASSIGN_VARIABLE();
TfLiteRegistration_V1 Register_AVERAGE_POOL_2D();
TfLiteRegistration_V1 Register_BATCH_TO_SPACE_ND();
TfLiteRegistration_V1 Register_BROADCAST_ARGS();
TfLiteRegistration_V1 Register_BROADCAST_TO();
TfLiteRegistration_V1 Register_CALL_ONCE();
TfLiteRegistration_V1 Register_CAST();
TfLiteRegistration_V1 Register_CEIL();
// TODO(b/160234179): Change custom OPs to also return by value.
TfLiteRegistration_V1* Register_CIRCULAR_BUFFER();
TfLiteRegistration_V1 Register_CONCATENATION();
TfLiteRegistration_V1 Register_CONV_2D();
TfLiteRegistration_V1 Register_COS();
TfLiteRegistration_V1 Register_CUMSUM();
TfLiteRegistration_V1 Register_DEPTH_TO_SPACE();
TfLiteRegistration_V1 Register_DEPTHWISE_CONV_2D();
TfLiteRegistration_V1 Register_DEQUANTIZE();
TfLiteRegistration_V1 Register_DIV();
TfLiteRegistration_V1 Register_ELU();
TfLiteRegistration_V1 Register_EQUAL();
TfLiteRegistration_V1* Register_ETHOSU();
TfLiteRegistration_V1 Register_EXP();
TfLiteRegistration_V1 Register_EXPAND_DIMS();
TfLiteRegistration_V1 Register_FILL();
TfLiteRegistration_V1 Register_FLOOR();
TfLiteRegistration_V1 Register_FLOOR_DIV();
TfLiteRegistration_V1 Register_FLOOR_MOD();
TfLiteRegistration_V1 Register_FULLY_CONNECTED();
TfLiteRegistration_V1 Register_GATHER();
TfLiteRegistration_V1 Register_GATHER_ND();
TfLiteRegistration_V1 Register_GREATER();
TfLiteRegistration_V1 Register_GREATER_EQUAL();
TfLiteRegistration_V1 Register_HARD_SWISH();
TfLiteRegistration_V1 Register_IF();
TfLiteRegistration_V1 Register_L2_NORMALIZATION();
TfLiteRegistration_V1 Register_L2_POOL_2D();
TfLiteRegistration_V1 Register_LEAKY_RELU();
TfLiteRegistration_V1 Register_LESS();
TfLiteRegistration_V1 Register_LESS_EQUAL();
TfLiteRegistration_V1 Register_LOG();
TfLiteRegistration_V1 Register_LOG_SOFTMAX();
TfLiteRegistration_V1 Register_LOGICAL_AND();
TfLiteRegistration_V1 Register_LOGICAL_NOT();
TfLiteRegistration_V1 Register_LOGICAL_OR();
TfLiteRegistration_V1 Register_LOGISTIC();
TfLiteRegistration_V1 Register_MAX_POOL_2D();
TfLiteRegistration_V1 Register_MAXIMUM();
TfLiteRegistration_V1 Register_MEAN();
TfLiteRegistration_V1 Register_MINIMUM();
TfLiteRegistration_V1 Register_MIRROR_PAD();
TfLiteRegistration_V1 Register_MUL();
TfLiteRegistration_V1 Register_NEG();
TfLiteRegistration_V1 Register_NOT_EQUAL();
TfLiteRegistration_V1 Register_PACK();
TfLiteRegistration_V1 Register_PAD();
TfLiteRegistration_V1 Register_PADV2();
TfLiteRegistration_V1 Register_PRELU();
TfLiteRegistration_V1 Register_QUANTIZE();
TfLiteRegistration_V1 Register_READ_VARIABLE();
TfLiteRegistration_V1 Register_REDUCE_MAX();
TfLiteRegistration_V1 Register_RELU();
TfLiteRegistration_V1 Register_RELU6();
TfLiteRegistration_V1 Register_RESIZE_BILINEAR();
TfLiteRegistration_V1 Register_RESIZE_NEAREST_NEIGHBOR();
TfLiteRegistration_V1 Register_RSQRT();
TfLiteRegistration_V1 Register_SELECT_V2();
TfLiteRegistration_V1 Register_SHAPE();
TfLiteRegistration_V1 Register_SIN();
TfLiteRegistration_V1 Register_SLICE();
TfLiteRegistration_V1 Register_SOFTMAX();
TfLiteRegistration_V1 Register_SPACE_TO_BATCH_ND();
TfLiteRegistration_V1 Register_SPACE_TO_DEPTH();
TfLiteRegistration_V1 Register_SPLIT();
TfLiteRegistration_V1 Register_SPLIT_V();
TfLiteRegistration_V1 Register_SQRT();
TfLiteRegistration_V1 Register_SQUARE();
TfLiteRegistration_V1 Register_SQUARED_DIFFERENCE();
TfLiteRegistration_V1 Register_SQUEEZE();
TfLiteRegistration_V1 Register_STRIDED_SLICE();
TfLiteRegistration_V1 Register_SUB();
TfLiteRegistration_V1 Register_SUM();
TfLiteRegistration_V1 Register_SVDF();
TfLiteRegistration_V1 Register_TANH();
TfLiteRegistration_V1 Register_TRANSPOSE();
TfLiteRegistration_V1 Register_TRANSPOSE_CONV();
// TODO(b/230666079): resolve conflict with xtensa implementation
TfLiteRegistration_V1 Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration_V1 Register_UNPACK();
TfLiteRegistration_V1 Register_VAR_HANDLE();
TfLiteRegistration_V1 Register_WHILE();
TfLiteRegistration_V1 Register_ZEROS_LIKE();

namespace ops {
namespace micro {
TfLiteRegistration_V1 Register_RESHAPE();
TfLiteRegistration_V1 Register_ROUND();
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_MICRO_OPS_H_
