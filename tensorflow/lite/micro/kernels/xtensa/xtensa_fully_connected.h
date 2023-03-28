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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_FULLY_CONNECTED_H_

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
struct XtensaFullyConnectedOpData {
  OpDataFullyConnected reference_op_data;

#if defined(VISION_P6)
  int8_t* reorder_coefficient_bias;  // buffers used to keep reordered coeff and
                                     // biases.
  uint32_t reorder_coefficient_bias_size;
  uint8_t* p_context;  // persistent lib context for this instance saved here
  uint32_t context_size;
#endif  // VISION_P6
};

#if defined(HIFIMINI)
void FullyConnectedEvalHifimini(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data);
#endif  // defined(HIFIMINI)

#if defined(VISION_P6)
TfLiteStatus FullyConnectedPrepareVision(TfLiteContext* context,
                                         TfLiteNode* node);

TfLiteStatus FullyConnectedEvalVision(TfLiteContext* context, TfLiteNode* node,
                                      const TfLiteConvParams& params,
                                      const XtensaFullyConnectedOpData& data,
                                      const TfLiteEvalTensor* input,
                                      const TfLiteEvalTensor* filter,
                                      const TfLiteEvalTensor* bias,
                                      TfLiteEvalTensor* output);
#endif  // VISION_P6

void* XtensaInitFullyConnected(TfLiteContext* context, const char* buffer,
                               size_t length);

TfLiteStatus XtensaEvalFullyConnectedQuantizedInt8(
    TfLiteContext* context, TfLiteNode* node, const OpDataFullyConnected& data,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output);

TfLiteStatus XtensaCalculateOpDataFullyConnected(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output, OpDataFullyConnected* data);

TfLiteStatus XtensaPrepareFullyConnected(TfLiteContext* context,
                                         TfLiteNode* node);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_FULLY_CONNECTED_H_
