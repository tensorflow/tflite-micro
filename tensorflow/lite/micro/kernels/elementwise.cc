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

#include "tensorflow/lite/micro/kernels/elementwise.h"

#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

TFLMRegistration Register_ABS() {
  return tflite::micro::RegisterOp(ElementWiseAbsRsqrtInit,
                                   ElementWiseAbsPrepare, AbsEvalReference);
}

TFLMRegistration Register_SIN() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseNumericPrepare,
                                   SinEvalReference);
}

TFLMRegistration Register_COS() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseNumericPrepare,
                                   CosEvalReference);
}

TFLMRegistration Register_LOG() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseNumericPrepare,
                                   LogEvalReference);
}

TFLMRegistration Register_SQRT() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseNumericPrepare,
                                   SqrtEvalReference);
}

TFLMRegistration Register_RSQRT() {
  return tflite::micro::RegisterOp(ElementWiseAbsRsqrtInit,
                                   ElementWiseRsqrtPrepare, RsqrtEvalReference);
}

TFLMRegistration Register_SQUARE() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseNumericPrepare,
                                   SquareEvalReference);
}

TFLMRegistration Register_LOGICAL_NOT() {
  return tflite::micro::RegisterOp(nullptr, ElementWiseLogicalPrepare,
                                   LogicalNotEvalReference);
}

}  // namespace micro
}  // namespace tflite
