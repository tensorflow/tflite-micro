/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/comparisons.h"

#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

TFLMRegistration Register_EQUAL() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   EqualEvalReference);
}

TFLMRegistration Register_NOT_EQUAL() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   NotEqualEvalReference);
}

TFLMRegistration Register_GREATER() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   GreaterEvalReference);
}

TFLMRegistration Register_GREATER_EQUAL() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   GreaterEqualEvalReference);
}

TFLMRegistration Register_LESS() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   LessEvalReference);
}

TFLMRegistration Register_LESS_EQUAL() {
  return tflite::micro::RegisterOp(ComparisonsInit, ComparisonsPrepare,
                                   LessEqualEvalReference);
}

}  // namespace micro
}  // namespace tflite
