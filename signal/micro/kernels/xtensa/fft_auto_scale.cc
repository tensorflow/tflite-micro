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

#include "tensorflow/lite/micro/audio_frontend/src/fft_auto_scale.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if XCHAL_HAVE_HIFI3
#include <xtensa/tie/xt_hifi3.h>
#endif

#include "tensorflow/lite/micro/audio_frontend/src/max_abs.h"
#include "tensorflow/lite/micro/audio_frontend/src/msb.h"

namespace tflite {
namespace tflm_signal {

#if XCHAL_HAVE_HIFI3
// Implementation for DSPs that support the Hifi3 ISA. Bit exact with the
// portable version below.
int FftAutoScale(const int16_t* input, int size, int16_t* output) {
  const int16_t max = MaxAbs16(input, size);
  int scale_bits = (sizeof(int16_t) * 8) - MostSignificantBit32(max) - 1;
  int i;
  if (scale_bits > 0) {
    const ae_int16x4* input_16x4_ptr =
        reinterpret_cast<const ae_int16x4*>(input);
    ae_int16x4* output_16x4_ptr = reinterpret_cast<ae_int16x4*>(output);
    const int num_iterations = ((size + 3) >> 2);
    for (i = 0; i < num_iterations; ++i) {
      ae_int16x4 input_16x4;
      AE_L16X4_IP(input_16x4, input_16x4_ptr, 8);
      ae_f16x4 input_f16x4 = *reinterpret_cast<ae_f16x4*>(&input_16x4);
      input_f16x4 = AE_SLAA16S(input_f16x4, scale_bits);
      input_16x4 = *reinterpret_cast<ae_int16x4*>(&input_f16x4);
      AE_S16X4_IP(input_16x4, output_16x4_ptr, 8);
    }
  } else {
    memcpy(output, input, size * sizeof(output[0]));
    scale_bits = 0;
  }
  return scale_bits;
}
#else
int FftAutoScale(const int16_t* input, int size, int16_t* output) {
  const int16_t max = MaxAbs16(input, size);
  int scale_bits = (sizeof(int16_t) * 8) - MostSignificantBit32(max) - 1;
  if (scale_bits <= 0) {
    scale_bits = 0;
  }
  for (int i = 0; i < size; i++) {
    // (input[i] << scale_bits) is undefined if input[i] is negative.
    // Multiply explicitly to make the code portable.
    output[i] = input[i] * (1 << scale_bits);
  }
  return scale_bits;
}
#endif

}  // namespace tflm_signal
}  // namespace tflite
