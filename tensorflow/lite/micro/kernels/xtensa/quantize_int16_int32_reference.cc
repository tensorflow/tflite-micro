/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/quantize.h"
#include "tensorflow/lite/micro/kernels/xtensa/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_quantize.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataQuantizeReference));
}

}  // namespace

TfLiteStatus EvalQuantizeInt16Int32Reference(TfLiteContext* context,
                                             TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* op_data = static_cast<OpDataQuantizeReference*>(node->user_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TFLITE_DCHECK((input->type == kTfLiteInt16 && output->type == kTfLiteInt32) ||
                (input->type == kTfLiteInt32 && output->type == kTfLiteInt32))
  int size = ElementCount(*input->dims);
  if (input->type == kTfLiteInt16) {
    reference_ops::Requantize(
        tflite::micro::GetTensorData<int16_t>(input), size,
        op_data->requantize_output_multiplier, op_data->requantize_output_shift,
        op_data->input_zero_point, op_data->quantization_params.zero_point,
        tflite::micro::GetTensorData<int32_t>(output));
  } else {
    reference_ops::Requantize(
        tflite::micro::GetTensorData<int32_t>(input), size,
        op_data->requantize_output_multiplier, op_data->requantize_output_shift,
        op_data->input_zero_point, op_data->quantization_params.zero_point,
        tflite::micro::GetTensorData<int16_t>(output));
  }
  return kTfLiteOk;
}

TfLiteRegistration Register_QUANTIZE_INT16_INT32REF() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/PrepareQuantizeReference,
          /*invoke=*/EvalQuantizeInt16Int32Reference,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
