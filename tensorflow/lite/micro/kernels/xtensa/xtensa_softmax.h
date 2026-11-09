/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SOFTMAX_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SOFTMAX_H_

#include <cstdint>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace tflite {
namespace micro {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
inline constexpr int kInt8SoftmaxLutSize = 256;
inline constexpr int kInt8SoftmaxLutThreshold = 256;

struct XtensaSoftmaxOpData {
  SoftmaxParams params;
  int scratch_tensor_index;
  int32_t* softmax_lut;
};
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

#if defined(VISION_P6)
struct XtensaSoftmaxOpData {
  SoftmaxParams params;
  uint8_t* p_context;  // persistent lib context for this instance saved here
  uint32_t context_size;
};
#endif  // defined(VISION_P6)

inline const SoftmaxParams& XtensaGetSoftmaxParams(const TfLiteNode* node) {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ) || \
    defined(VISION_P6)
  return static_cast<const XtensaSoftmaxOpData*>(node->user_data)->params;
#else
  return *static_cast<const SoftmaxParams*>(node->user_data);
#endif
}

void* XtensaInitSoftmax(TfLiteContext* context, const char* buffer,
                        size_t length);

TfLiteStatus XtensaPrepareSoftmax(TfLiteContext* context, TfLiteNode* node);

template <TfLiteType kInputType, TfLiteType kOutputType>
TfLiteStatus XtensaPrepareSoftmaxTyped(TfLiteContext* context,
                                       TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kSoftmaxInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kSoftmaxOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kInputType);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kOutputType);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return XtensaPrepareSoftmax(context, node);
}

TfLiteStatus XtensaEvalSoftmaxInt8Int16(TfLiteContext* context,
                                        TfLiteNode* node);

#if defined(VISION_P6)
TfLiteStatus SoftmaxPrepareVision(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus SoftmaxEvalVision(TfLiteContext* context, TfLiteNode* node,
                               const XtensaSoftmaxOpData& data,
                               const TfLiteEvalTensor* input,
                               TfLiteEvalTensor* output);
#endif  // defined(VISION_P6)

}  // namespace micro

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ) || \
    defined(VISION_P6)
using micro::XtensaSoftmaxOpData;
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ) || defined(VISION_P6)
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SOFTMAX_H_
