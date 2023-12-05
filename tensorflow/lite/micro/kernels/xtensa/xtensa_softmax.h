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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/softmax.h"

namespace tflite {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
struct XtensaSoftmaxOpData {
  SoftmaxParams params;
  int scratch_tensor_index;
};
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

#if defined(VISION_P6)
struct XtensaSoftmaxOpData {
  SoftmaxParams params;
  uint8_t* p_context;  // persistent lib context for this instance saved here
  uint32_t context_size;
};
#endif  // defined(VISION_P6)

void* XtensaInitSoftmax(TfLiteContext* context, const char* buffer,
                        size_t length);

TfLiteStatus XtensaPrepareSoftmax(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus XtensaEvalSoftmaxInt8Int16(TfLiteContext* context,
                                        TfLiteNode* node);

#if defined(VISION_P6)
TfLiteStatus SoftmaxPrepareVision(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus SoftmaxEvalVision(TfLiteContext* context, TfLiteNode* node,
                               const XtensaSoftmaxOpData& data,
                               const TfLiteEvalTensor* input,
                               TfLiteEvalTensor* output);
#endif  // defined(VISION_P6)

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SOFTMAX_H_
