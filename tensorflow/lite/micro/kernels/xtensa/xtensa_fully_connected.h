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

template <TfLiteType T = kTfLiteVariant>
TfLiteStatus XtensaPrepareFullyConnected(TfLiteContext* context,
                                         TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);
  const auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (kTfLiteInt8 == T) {
    if (input->type != kTfLiteInt8) {
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
    }
  }

  if (filter->type == kTfLiteInt4) {
    int filter_size =
        RuntimeShape(filter->dims->size,
                     reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(context, filter_size,
                                         &data->filter_buffer_index);
  }

  TFLITE_DCHECK(GetTensorShape(output).DimensionsCount() == 2);

  TF_LITE_ENSURE_OK(context, CalculateOpDataFullyConnected(
                                 context, params->activation, input->type,
                                 input, filter, bias, output, data));

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);
#if defined(VISION_P6)
  TF_LITE_ENSURE_OK(context, FullyConnectedPrepareVision(context, node));
#endif  // VISION_P6

  return kTfLiteOk;
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_FULLY_CONNECTED_H_
