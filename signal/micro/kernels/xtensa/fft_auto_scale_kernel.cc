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

#include "signal/micro/kernels/fft_auto_scale_kernel.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "signal/src/fft_auto_scale.h"
#include "signal/src/max_abs.h"
#include "signal/src/msb.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

#if XCHAL_HAVE_HIFI3
#include <xtensa/tie/xt_hifi3.h>
namespace {
// Implementation for DSPs that support the Hifi3 ISA. Bit exact with the
// portable version below.
int XtensaFftAutoScale(const int16_t* input, int size, int16_t* output) {
  const int16_t max = tflite::tflm_signal::MaxAbs16(input, size);
  int scale_bits = (sizeof(int16_t) * 8) -
                   tflite::tflm_signal::MostSignificantBit32(max) - 1;
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
}  // namespace
#endif

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr int kScaleBitTensor = 1;

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TfLiteEvalTensor* scale_bit =
      tflite::micro::GetEvalOutput(context, node, kScaleBitTensor);

  const int16_t* input_data = tflite::micro::GetTensorData<int16_t>(input);
  int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);
  int32_t* scale_bit_data = tflite::micro::GetTensorData<int32_t>(scale_bit);

#if XCHAL_HAVE_HIFI3
  *scale_bit_data =
      XtensaFftAutoScale(input_data, output->dims->data[0], output_data);
#else
  *scale_bit_data =
      tflm_signal::FftAutoScale(input_data, output->dims->data[0], output_data);
#endif
  return kTfLiteOk;
}

}  // namespace

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflm_signal {

TFLMRegistration* Register_FFT_AUTO_SCALE() {
  static TFLMRegistration r =
      tflite::micro::RegisterOp(nullptr, FftAutoScalePrepare, Eval);
  return &r;
}

}  // namespace tflm_signal
}  // namespace tflite
