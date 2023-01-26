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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_fully_connected.h"

namespace tflite {

void* XtensaInitFullyConnected(TfLiteContext* context, const char* buffer,
                               size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if !defined(VISION_P6)
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataFullyConnected));
#else
  void* data = context->AllocatePersistentBuffer(
      context, sizeof(XtensaFullyConnectedOpData));
#if !defined(HIFIMINI)
  if (InitXtensaContext()) {
    return nullptr;
  }
#endif
  return data;
#endif  // defined(VISION_P6)
}

TfLiteStatus XtensaCalculateOpDataFullyConnected(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    OpDataFullyConnected* data) {
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
#if defined(HIFIMINI)
    if (input->type == kTfLiteInt8) {
      QuantizeMultiplierForInt24(real_multiplier, &data->output_multiplier,
                                 &data->output_shift);
    } else {
      QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                         &data->output_shift);
    }
#else
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
#endif

    // Filter weights will always be symmetric quantized since we only support
    // int8 quantization. See
    // https://github.com/tensorflow/tensorflow/issues/44912 for additional
    // context.
    TFLITE_DCHECK(filter->params.zero_point == 0);

    data->input_zero_point = input->params.zero_point;
    data->filter_zero_point = filter->params.zero_point;
    data->output_zero_point = output->params.zero_point;

    return CalculateActivationRangeQuantized(context, activation, output,
                                             &data->output_activation_min,
                                             &data->output_activation_max);
  }
  return kTfLiteOk;
}

}  // namespace tflite
