/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_BATCH_MATMUL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_BATCH_MATMUL_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {

struct QuantizationOpDataBatchMatmul {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;  // exponent

  // The range of the fused activation layer. For example for kNone and
  // int8_t these would be -128 and 127.
  int32_t output_activation_min;
  int32_t output_activation_max;

  int32_t lhs_zero_point;
  int32_t rhs_zero_point;
  int32_t output_zero_point;
};

struct OpDataBatchMatmul {
  QuantizationOpDataBatchMatmul* quantization;

  // Transpose tensors and state
  TfLiteEvalTensor* lhs_transposed_tensor;
  TfLiteEvalTensor* rhs_transposed_tensor;
  bool rhs_is_transposed;
  bool lhs_is_constant_tensor;
  bool rhs_is_constant_tensor;
};

extern const int kBatchMatmulInputLhsTensor;
extern const int kBatchMatmulInputRhsTensor;
extern const int kBatchMatmulOutputTensor;

TfLiteStatus ReshapeOutputTensor(TfLiteContext* context, TfLiteNode* node,
                                 const RuntimeShape& extended_lhs_shape,
                                 const RuntimeShape& extended_rhs_shape,
                                 bool adj_x, bool adj_y, int output_rank,
                                 TfLiteTensor* output);

template <typename T>
void TransposeRowsColumnsImpl(const TfLiteEvalTensor& tensor_in,
                              TfLiteEvalTensor* tensor_out);

TfLiteStatus TransposeRowsColumns(const TfLiteEvalTensor& tensor_in,
                                  TfLiteEvalTensor* tensor_out);

RuntimeShape SwapRowColumnDims(const RuntimeShape& shape);

TFLMRegistration Register_BATCH_MATMUL();

#if defined(CMSIS_NN)

// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 matrix multiplication and uses the latency optimized
// implementations.
TFLMRegistration Register_BATCH_MATMUL_INT8();

// Returns a TFLMRegistration struct for kernel variant that only supports
// int16 matrix multiplication and uses the latency optimized
// implementations.
TFLMRegistration Register_BATCH_MATMUL_INT16();

#else
inline TFLMRegistration Register_BATCH_MATMUL_INT8() {
  return Register_BATCH_MATMUL();
}

inline TFLMRegistration Register_BATCH_MATMUL_INT16() {
  return Register_BATCH_MATMUL();
}
#endif  // defined(CMSIS_NN)

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_BATCH_MATMUL_H_
