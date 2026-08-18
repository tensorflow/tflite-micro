/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "signal/micro/kernels/irfft.h"
#include "signal/micro/kernels/rfft.h"
#include "tensorflow/lite/c/common.h"

// Forward declaration of all micro op kernel registration methods. These
// registrations are included with the standard `BuiltinOpResolver`.
//
// This header is particularly useful in cases where only a subset of ops are
// needed. In such cases, the client can selectively add only the registrations
// their model requires, using a custom `(Micro)MutableOpResolver`. Selective
// registration in turn allows the linker to strip unused kernels.

namespace tflite {
namespace micro {
// TFLM is incrementally moving towards a flat tflite namespace
// (https://abseil.io/tips/130). Any new ops (or cleanup of existing ops should
// have their Register function declarations in the tflite namespace.

TFLMRegistration Register_ABS();
TFLMRegistration Register_ADD();
TFLMRegistration Register_ADD_N();
TFLMRegistration Register_ARG_MAX();
TFLMRegistration Register_ARG_MIN();
TFLMRegistration Register_ASSIGN_VARIABLE();
TFLMRegistration Register_AVERAGE_POOL_2D();
TFLMRegistration Register_BATCH_MATMUL();
TFLMRegistration Register_BATCH_TO_SPACE_ND();
TFLMRegistration Register_BROADCAST_ARGS();
TFLMRegistration Register_BROADCAST_TO();
TFLMRegistration Register_CALL_ONCE();
TFLMRegistration Register_CAST();
TFLMRegistration Register_CEIL();
// TODO(b/160234179): Change custom OPs to also return by value.
TFLMRegistration* Register_CIRCULAR_BUFFER();
TFLMRegistration Register_CONCATENATION();
TFLMRegistration Register_CONV_2D();
TFLMRegistration Register_COS();
TFLMRegistration Register_CUMSUM();
TFLMRegistration Register_DECODE();
TFLMRegistration Register_DEPTH_TO_SPACE();
TFLMRegistration Register_DEPTHWISE_CONV_2D();
TFLMRegistration Register_DEQUANTIZE();
TFLMRegistration Register_DIV();
TFLMRegistration Register_DYNAMIC_UPDATE_SLICE();
TFLMRegistration Register_ELU();
TFLMRegistration Register_EMBEDDING_LOOKUP();
TFLMRegistration Register_EQUAL();
TFLMRegistration* Register_ETHOSU();
TFLMRegistration Register_EXP();
TFLMRegistration Register_EXPAND_DIMS();
TFLMRegistration Register_FILL();
TFLMRegistration Register_FLOOR();
TFLMRegistration Register_FLOOR_DIV();
TFLMRegistration Register_FLOOR_MOD();
TFLMRegistration Register_FULLY_CONNECTED();
TFLMRegistration Register_GATHER();
TFLMRegistration Register_GATHER_ND();
TFLMRegistration Register_GREATER();
TFLMRegistration Register_GREATER_EQUAL();
TFLMRegistration Register_HARD_SWISH();
TFLMRegistration Register_IF();
TFLMRegistration Register_L2_NORMALIZATION();
TFLMRegistration Register_L2_POOL_2D();
TFLMRegistration Register_LEAKY_RELU();
TFLMRegistration Register_LESS();
TFLMRegistration Register_LESS_EQUAL();
TFLMRegistration Register_LOG();
TFLMRegistration Register_LOG_SOFTMAX();
TFLMRegistration Register_LOGICAL_AND();
TFLMRegistration Register_LOGICAL_NOT();
TFLMRegistration Register_LOGICAL_OR();
TFLMRegistration Register_LOGISTIC();
TFLMRegistration Register_MAX_POOL_2D();
TFLMRegistration Register_MAXIMUM();
TFLMRegistration Register_MEAN();
TFLMRegistration Register_MINIMUM();
TFLMRegistration Register_MIRROR_PAD();
TFLMRegistration Register_MUL();
TFLMRegistration Register_NEG();
TFLMRegistration Register_NOT_EQUAL();
TFLMRegistration Register_PACK();
TFLMRegistration Register_PAD();
TFLMRegistration Register_PADV2();
TFLMRegistration Register_PRELU();
TFLMRegistration Register_QUANTIZE();
TFLMRegistration Register_READ_VARIABLE();
TFLMRegistration Register_REDUCE_ALL();
TFLMRegistration Register_REDUCE_MAX();
TFLMRegistration Register_REDUCE_MIN();
TFLMRegistration Register_RELU();
TFLMRegistration Register_RELU6();
TFLMRegistration Register_RESHAPE();
TFLMRegistration Register_RESIZE_BILINEAR();
TFLMRegistration Register_RESIZE_NEAREST_NEIGHBOR();
TFLMRegistration Register_REVERSE_V2();
TFLMRegistration Register_ROUND();
TFLMRegistration Register_RSQRT();
TFLMRegistration Register_SELECT_V2();
TFLMRegistration Register_SHAPE();
TFLMRegistration Register_SIN();
TFLMRegistration Register_SLICE();
TFLMRegistration Register_SOFTMAX();
TFLMRegistration Register_SPACE_TO_BATCH_ND();
TFLMRegistration Register_SPACE_TO_DEPTH();
TFLMRegistration Register_SPLIT();
TFLMRegistration Register_SPLIT_V();
TFLMRegistration Register_SQRT();
TFLMRegistration Register_SQUARE();
TFLMRegistration Register_SQUARED_DIFFERENCE();
TFLMRegistration Register_SQUEEZE();
TFLMRegistration Register_STRIDED_SLICE();
TFLMRegistration Register_SUB();
TFLMRegistration Register_SUM();
TFLMRegistration Register_SVDF();
TFLMRegistration Register_TANH();
TFLMRegistration Register_TRANSPOSE();
TFLMRegistration Register_TRANSPOSE_CONV();
// TODO(b/230666079): resolve conflict with xtensa implementation
TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
TFLMRegistration Register_UNPACK();
TFLMRegistration Register_VAR_HANDLE();
TFLMRegistration Register_WHILE();
TFLMRegistration Register_ZEROS_LIKE();

// TODO(b/295174388): Add the rest of inference only registration functions.
TFLMInferenceRegistration RegisterInference_FULLY_CONNECTED();

// TODO(b/160234179): Change custom OPs to also return by value.
namespace tflm_signal {
TFLMRegistration* Register_DELAY();
TFLMRegistration* Register_FFT_AUTO_SCALE();
TFLMRegistration* Register_FILTER_BANK();
TFLMRegistration* Register_FILTER_BANK_LOG();
TFLMRegistration* Register_FILTER_BANK_SPECTRAL_SUBTRACTION();
TFLMRegistration* Register_FILTER_BANK_SQUARE_ROOT();
TFLMRegistration* Register_ENERGY();
TFLMRegistration* Register_FRAMER();
TFLMRegistration* Register_OVERLAP_ADD();
TFLMRegistration* Register_PCAN();
TFLMRegistration* Register_STACKER();
TFLMRegistration* Register_WINDOW();
}  // namespace tflm_signal

}  // namespace micro

namespace tflm_signal {
using micro::tflm_signal::Register_DELAY;
using micro::tflm_signal::Register_ENERGY;
using micro::tflm_signal::Register_FFT_AUTO_SCALE;
using micro::tflm_signal::Register_FILTER_BANK;
using micro::tflm_signal::Register_FILTER_BANK_LOG;
using micro::tflm_signal::Register_FILTER_BANK_SPECTRAL_SUBTRACTION;
using micro::tflm_signal::Register_FILTER_BANK_SQUARE_ROOT;
using micro::tflm_signal::Register_FRAMER;
using micro::tflm_signal::Register_OVERLAP_ADD;
using micro::tflm_signal::Register_PCAN;
using micro::tflm_signal::Register_STACKER;
using micro::tflm_signal::Register_WINDOW;
}  // namespace tflm_signal
using micro::Register_ABS;
using micro::Register_ADD;
using micro::Register_ADD_N;
using micro::Register_ARG_MAX;
using micro::Register_ARG_MIN;
using micro::Register_ASSIGN_VARIABLE;
using micro::Register_AVERAGE_POOL_2D;
using micro::Register_BATCH_MATMUL;
using micro::Register_BATCH_TO_SPACE_ND;
using micro::Register_BROADCAST_ARGS;
using micro::Register_BROADCAST_TO;
using micro::Register_CALL_ONCE;
using micro::Register_CAST;
using micro::Register_CEIL;
using micro::Register_CIRCULAR_BUFFER;
using micro::Register_CONCATENATION;
using micro::Register_CONV_2D;
using micro::Register_COS;
using micro::Register_CUMSUM;
using micro::Register_DECODE;
using micro::Register_DEPTH_TO_SPACE;
using micro::Register_DEPTHWISE_CONV_2D;
using micro::Register_DEQUANTIZE;
using micro::Register_DIV;
using micro::Register_DYNAMIC_UPDATE_SLICE;
using micro::Register_ELU;
using micro::Register_EMBEDDING_LOOKUP;
using micro::Register_EQUAL;
using micro::Register_ETHOSU;
using micro::Register_EXP;
using micro::Register_EXPAND_DIMS;
using micro::Register_FILL;
using micro::Register_FLOOR;
using micro::Register_FLOOR_DIV;
using micro::Register_FLOOR_MOD;
using micro::Register_FULLY_CONNECTED;
using micro::Register_GATHER;
using micro::Register_GATHER_ND;
using micro::Register_GREATER;
using micro::Register_GREATER_EQUAL;
using micro::Register_HARD_SWISH;
using micro::Register_IF;
using micro::Register_L2_NORMALIZATION;
using micro::Register_L2_POOL_2D;
using micro::Register_LEAKY_RELU;
using micro::Register_LESS;
using micro::Register_LESS_EQUAL;
using micro::Register_LOG;
using micro::Register_LOG_SOFTMAX;
using micro::Register_LOGICAL_AND;
using micro::Register_LOGICAL_NOT;
using micro::Register_LOGICAL_OR;
using micro::Register_LOGISTIC;
using micro::Register_MAX_POOL_2D;
using micro::Register_MAXIMUM;
using micro::Register_MEAN;
using micro::Register_MINIMUM;
using micro::Register_MIRROR_PAD;
using micro::Register_MUL;
using micro::Register_NEG;
using micro::Register_NOT_EQUAL;
using micro::Register_PACK;
using micro::Register_PAD;
using micro::Register_PADV2;
using micro::Register_PRELU;
using micro::Register_QUANTIZE;
using micro::Register_READ_VARIABLE;
using micro::Register_REDUCE_ALL;
using micro::Register_REDUCE_MAX;
using micro::Register_REDUCE_MIN;
using micro::Register_RELU;
using micro::Register_RELU6;
using micro::Register_RESHAPE;
using micro::Register_RESIZE_BILINEAR;
using micro::Register_RESIZE_NEAREST_NEIGHBOR;
using micro::Register_REVERSE_V2;
using micro::Register_ROUND;
using micro::Register_RSQRT;
using micro::Register_SELECT_V2;
using micro::Register_SHAPE;
using micro::Register_SIN;
using micro::Register_SLICE;
using micro::Register_SOFTMAX;
using micro::Register_SPACE_TO_BATCH_ND;
using micro::Register_SPACE_TO_DEPTH;
using micro::Register_SPLIT;
using micro::Register_SPLIT_V;
using micro::Register_SQRT;
using micro::Register_SQUARE;
using micro::Register_SQUARED_DIFFERENCE;
using micro::Register_SQUEEZE;
using micro::Register_STRIDED_SLICE;
using micro::Register_SUB;
using micro::Register_SUM;
using micro::Register_SVDF;
using micro::Register_TANH;
using micro::Register_TRANSPOSE;
using micro::Register_TRANSPOSE_CONV;
using micro::Register_UNIDIRECTIONAL_SEQUENCE_LSTM;
using micro::Register_UNPACK;
using micro::Register_VAR_HANDLE;
using micro::Register_WHILE;
using micro::Register_ZEROS_LIKE;
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_MICRO_OPS_H_
