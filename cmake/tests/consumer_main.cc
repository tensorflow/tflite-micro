// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "models/simple_add_model_model_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#ifdef TFLM_SMOKE_TEST_SIGNAL
#include "signal/micro/kernels/rfft.h"
#endif

int main() {
  const tflite::Model* model = tflite::GetModel(g_simple_add_model_model_data);
  if (model == nullptr) {
    return 1;
  }

  tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddAdd() != kTfLiteOk) {
    return 2;
  }
#ifdef TFLM_SMOKE_TEST_SIGNAL
  if (resolver.AddRfft() != kTfLiteOk) {
    return 3;
  }
#endif

  alignas(16) static uint8_t arena[128 * 1024];
  tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 4;
  }

  for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
    TfLiteTensor* input = interpreter.input(i);
    if (input == nullptr || input->data.raw == nullptr) {
      return 5;
    }
    std::memset(input->data.raw, 0, input->bytes);
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    return 6;
  }

  const TfLiteTensor* output = interpreter.output(0);
  if (output == nullptr || output->bytes == 0) {
    return 7;
  }
  return 0;
}
