/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_QUANTIZE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_QUANTIZE_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

struct OpDataQuantizeReference {
  tflite::QuantizationParams quantization_params;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t requantize_output_multiplier;
  int requantize_output_shift;

  int32_t input_zero_point;
};

TfLiteStatus EvalQuantizeReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus PrepareQuantizeReference(TfLiteContext* context, TfLiteNode* node);

// This is the most generic TfLiteRegistration. The actual supported types may
// still be target dependent. The only requirement is that every implementation
// (reference or optimized) must define this function.
TfLiteRegistration Register_QUANTIZE();

#if defined(XTENSA)
// Returns a TfLiteRegistration struct for kernel variant that only supports
// int8 inputs and outputs.
TfLiteRegistration Register_QUANTIZE_INT16_INT32REF();

#else
inline TfLiteRegistration Register_QUANTIZE_INT16_INT32REF() {
  return Register_QUANTIZE();
}
#endif
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_QUANTIZE_H_
