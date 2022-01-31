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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/dequantize.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

TfLiteStatus DequantizePrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  DequantizeOpData* data = static_cast<DequantizeOpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // TODO(b/140515557): Add cached dequant to improve hybrid model performance.
  const TfLiteTensor* input = AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context,
                 input->type == kTfLiteInt8 || input->type == kTfLiteInt16);
  TF_LITE_ENSURE(context, output->type == kTfLiteFloat32);

  if (output->type == kTfLiteInt32) {
    const double effective_output_scale =
        static_cast<double>(input->params.scale) /
        static_cast<double>(output->params.scale);
    QuantizeMultiplier(effective_output_scale, &data->output_multiplier,
                       &data->output_shift);
  }

  data->quantization_params.zero_point = input->params.zero_point;
  data->quantization_params.scale = static_cast<double>(input->params.scale);
  data->output_zero_point = output->params.zero_point;
  return kTfLiteOk;
}

}  // namespace tflite
