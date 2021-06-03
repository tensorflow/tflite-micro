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

#include "tensorflow/lite/micro/kernels/quantize.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataQuantizeReference));
}

TfLiteStatus PrepareQuantizeInt8(TfLiteContext* context,
                                      TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* data = static_cast<OpDataQuantizeReference*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  double effective_scale = static_cast<double>(input->params.scale) /
                           static_cast<double>(output->params.scale);

  QuantizeMultiplier(effective_scale, &data->requantize_output_multiplier,
                     &data->requantize_output_shift);

  data->quantization_params.zero_point = output->params.zero_point;
  data->quantization_params.scale = static_cast<double>(output->params.scale);

  data->input_zero_point = input->params.zero_point;
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizeInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* data = static_cast<OpDataQuantizeReference*>(node->user_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  if (input->type == kTfLiteInt16) {
    size_t size = ElementCount(*input->dims);
      reference_ops::Requantize(
          tflite::micro::GetTensorData<int16_t>(input), size,
          data->requantize_output_multiplier, data->requantize_output_shift,
          data->input_zero_point, data->quantization_params.zero_point,
          tflite::micro::GetTensorData<int8_t>(output));
  } else if (input->type == kTfLiteInt8) {
    // Int8 to Int8 requantization, required if the input and output tensors
    // have different scales and/or zero points.
    size_t size = ElementCount(*input->dims);
    reference_ops::Requantize(
        tflite::micro::GetTensorData<int8_t>(input), size,
        data->requantize_output_multiplier, data->requantize_output_shift,
        data->input_zero_point, data->quantization_params.zero_point,
        tflite::micro::GetTensorData<int32_t>(output));
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_QUANTIZE() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/PrepareQuantizeReference,
          /*invoke=*/EvalQuantizeReference,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_QUANTIZE_INT8() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/PrepareQuantizeInt8,
          /*invoke=*/EvalQuantizeInt8,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
