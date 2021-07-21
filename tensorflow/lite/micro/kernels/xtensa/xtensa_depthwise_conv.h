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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_DEPTHWISE_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_DEPTHWISE_CONV_H_

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"

namespace tflite {
struct XtensaDepthwiseConvOpData {
  OpDataConv reference_op_data;

#if defined(HIFI4) || defined(HIFI5)
  int scratch_tensor_index;
#endif  // defined(HIFI4) || defined(HIFI5)
};

#if defined(HIFIMINI)
void DepthwiseConvEvalHifiMini(
    const DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data);

inline void DepthwiseConv4x32MatchingInputAndFilterHifiMini(
    const int input_offset, const int output_offset,
    const int quantized_activation_min, const int quantized_activation_max,
    const int32_t* output_multiplier, const int32_t* output_shift,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data);

#elif defined(HIFI4) || defined(HIFI5)
TfLiteStatus DepthwiseConvPrepareHifi(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus DepthwiseConvEvalHifi(TfLiteContext* context, TfLiteNode* node,
                                   const TfLiteDepthwiseConvParams& params,
                                   const XtensaDepthwiseConvOpData& data,
                                   const TfLiteEvalTensor* input,
                                   const TfLiteEvalTensor* filter,
                                   const TfLiteEvalTensor* bias,
                                   TfLiteEvalTensor* output);

TfLiteStatus DepthwiseConvReferenceEvalInt8(TfLiteContext* context,
                                            TfLiteNode* node);
#endif

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_DEPTHWISE_CONV_H_
