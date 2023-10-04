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

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  int scratch_tensor_index;
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

#if defined(VISION_P6)
  int8_t* reorder_coefficient_bias;  // buffers used to keep reordered coeff and
                                     // biases.
  uint32_t reorder_coefficient_bias_size;
  int8_t* per_channel_output_shift_int8;
  uint8_t* p_context;  // persistent lib context for this instance saved here
  uint32_t context_size;
#endif  // VISION_P6
};

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
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
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

#if defined(VISION_P6)

TfLiteStatus DepthwiseConvPrepareVision(TfLiteContext* context,
                                        TfLiteNode* node);

TfLiteStatus DepthwiseConvEvalVision(TfLiteContext* context, TfLiteNode* node,
                                     const TfLiteDepthwiseConvParams& params,
                                     const XtensaDepthwiseConvOpData& data,
                                     const TfLiteEvalTensor* input,
                                     const TfLiteEvalTensor* filter,
                                     const TfLiteEvalTensor* bias,
                                     TfLiteEvalTensor* output);

#endif  // VISION_P6

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_DEPTHWISE_CONV_H_
