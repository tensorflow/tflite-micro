/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "signal/src/energy.h"

#include <stddef.h>
#include <string.h>

#include "signal/src/complex.h"

namespace tflite {
namespace tflm_signal {
void SpectrumToEnergy(const Complex<int16_t>* input, int start_index,
                      int end_index, int output_size, uint32_t* output) {
  if (output_size <= 0) {
    return;
  }
  const int start =
      start_index < 0 ? 0
                      : (start_index > output_size ? output_size : start_index);
  const int end = end_index < start
                      ? start
                      : (end_index > output_size ? output_size : end_index);
  if (start > 0) {
    memset(output, 0, static_cast<size_t>(start) * sizeof(uint32_t));
  }
  for (int i = start; i < end; i++) {
    const int32_t real = input[i].real;
    const int32_t imag = input[i].imag;
    output[i] =
        static_cast<uint32_t>(real * real) + static_cast<uint32_t>(imag * imag);
  }
  if (end < output_size) {
    memset(output + end, 0,
           static_cast<size_t>(output_size - end) * sizeof(uint32_t));
  }
}

}  // namespace tflm_signal
}  // namespace tflite
