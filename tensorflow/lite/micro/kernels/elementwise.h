/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_ELEMENTWISE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_ELEMENTWISE_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

extern const int kElementwiseInputTensor;
extern const int kElementwiseOutputTensor;

struct OpDataAbsRsqrt {
  int32_t multiplier;
  int shift;
  int input_offset;
  int output_offset;
  bool needs_rescale;
  TfLiteQuantizationType input_quantization_type;
  TfLiteType input_type;
};

void* ElementWiseAbsRsqrtInit(TfLiteContext* context, const char* buffer,
                              size_t length);

TfLiteStatus ElementWiseNumericPrepare(TfLiteContext* context,
                                       TfLiteNode* node);
TfLiteStatus ElementWiseLogicalPrepare(TfLiteContext* context,
                                       TfLiteNode* node);
TfLiteStatus ElementWiseAbsPrepare(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus ElementWiseRsqrtPrepare(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus AbsEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus SinEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus CosEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus LogEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus SqrtEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus RsqrtEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus SquareEvalReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus LogicalNotEvalReference(TfLiteContext* context, TfLiteNode* node);

TFLMRegistration Register_ABS();
TFLMRegistration Register_SIN();
TFLMRegistration Register_COS();
TFLMRegistration Register_LOG();
TFLMRegistration Register_SQRT();
TFLMRegistration Register_RSQRT();
TFLMRegistration Register_SQUARE();
TFLMRegistration Register_LOGICAL_NOT();

}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ELEMENTWISE_H_
