/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_CONV_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

struct OpDataTransposeConv {
  ConvParams params;

  // A scratch buffer is required for quantized implementations.
  int scratch_buffer_index;

  // TODO(b/192090531): Remove this once all 8x16 transpose conv models use
  // 64-bit biases.
  int bias_converted_buffer_index;

  // Multiplier and shift arrays are required for the int8 implementation.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;
};

// For the TfLite transpose_conv implementation, input tensor 0 corresponds to
// the OutputShapeTensor. However, since TFLM does not support dynamic tensors,
// the TFLM implementation ignores input tensor 0 and the only inputs we care
// about are kFilterTensor, kInputTensor and kBiasTensor.
extern const int kTransposeConvFilterTensor;
extern const int kTransposeConvInputTensor;
extern const int kTransposeConvBiasTensor;
extern const int kTransposeConvOutputTensor;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
extern const int kTransposeConvQuantizedDimension;

TfLiteStatus CalculateOpDataTransposeConv(
    TfLiteContext* context, TfLiteNode* node,
    const TfLiteTransposeConvParams& params, int width, int height,
    int filter_width, int filter_height, int out_width, int out_height,
    const TfLiteType data_type, OpDataTransposeConv* data);

TfLiteStatus TransposeConvPrepare(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus TransposeConvEvalInt16x8Reference(TfLiteContext* context,
                                               TfLiteNode* node);

// This is the most generic TfLiteRegistration. The actual supported types may
// still be target dependent. The only requirement is that every implementation
// (reference or optimized) must define this function.
TfLiteRegistration Register_TRANSPOSE_CONV();

#if defined(XTENSA)
// Returns a TfLiteRegistration struct for kernel variant that only supports
// int16 inputs and outputs and int8 weights.
TfLiteRegistration Register_TRANSPOSE_CONV_INT16X8REF();

#else
inline TfLiteRegistration Register_TRANSPOSE_CONV_INT16X8REF() {
  return Register_TRANSPOSE_CONV();
}
#endif

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_CONV_H_
