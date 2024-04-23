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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_STREAMING_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_STREAMING_CONV_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {

struct OpDataSConv {
  TfLitePaddingValues padding;

  // Cached tensor zero point values for quantized operations.
  int32_t input_zero_point;
  int32_t filter_zero_point;
  int32_t output_zero_point;

  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  // A buffer used to store unpacked filter values. This is used if the source
  // tensor is of n-bit precision that cannot be easily processed by kernels.
  int filter_buffer_index;
};

struct OpDataStreamingConv {
  struct OpDataSConv op_data;
  void* input_state;
};

extern const int kConvInputTensor;
extern const int kConvWeightsTensor;
extern const int kConvBiasTensor;
extern const int kConvOutputTensor;
extern const int kConvQuantizedDimension;

ConvParams StreamingConvParamsQuantized(const TfLiteConvParams& params,
                                        const OpDataSConv& data);

void* StreamingConvInit(TfLiteContext* context, const char* buffer,
                        size_t length);
TfLiteStatus StreamingConvPrepare(TfLiteContext* context, TfLiteNode* node);

// This is the most generic TFLMRegistration. The actual supported types
// may still be target dependent. The only requirement is that every
// implementation (reference or optimized) must define this function.
TFLMRegistration Register_STREAMING_CONV_2D();

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_STREAMING_CONV_H_
