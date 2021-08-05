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

#include "tensorflow/lite/kernels/internal/reference/reduce.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

TfLiteStatus PrepareSimple(TfLiteContext* context, TfLiteNode* node) {
  // Inputs Tensor (dtype depends on quantization):
  // [0] = Input
  // [1] = Axis
  const TfLiteTensor* input = GetInput(context, node, 0);

  // Outputs Tensor (dtype depends on quantization):
  // [0] = Output

  // Validate number of inputs and outputs
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Validate axis type
  const TfLiteTensor* axis = GetInput(context, node, 1);
  TF_LITE_ENSURE(context, axis != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, axis->type, kTfLiteInt32);

  if (input->type == kTfLiteInt8) {
    OpDataReduce* data = static_cast<OpDataReduce*>(node->user_data);
    const TfLiteTensor* output = GetOutput(context, node, 0);
    const double real_multiplier = static_cast<double>(input->params.scale) /
                                   static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, &data->multiplier, &data->shift);
  }

  return kTfLiteOk;
}

TfLiteStatus PrepareMax(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node));

  OpDataReduce* op_data = static_cast<OpDataReduce*>(node->user_data);
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* axis = GetInput(context, node, 1);

  op_data->input_scale = input->params.scale;
  op_data->output_scale = output->params.scale;
  op_data->num_output_elements = NumElements(output);

  context->RequestScratchBufferInArena(context, sizeof(int) * input->dims->size,
                                       &op_data->temp_buffer_idx);
  context->RequestScratchBufferInArena(
      context, sizeof(int) * static_cast<int>(ElementCount(*axis->dims)),
      &op_data->resolved_axis_idx);

  return kTfLiteOk;
}

TfLiteStatus PrepareMeanOrSum(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  OpDataReduce* op_data = reinterpret_cast<OpDataReduce*>(node->user_data);
  const TfLiteTensor* output = GetOutput(context, node, 0);
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    const double real_multiplier = static_cast<double>(input->params.scale) /
                                   static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, &op_data->multiplier, &op_data->shift);
  }

  int output_size = NumElements(output);
  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8 ||
      input->type == kTfLiteInt16) {
    context->RequestScratchBufferInArena(context, output_size * sizeof(int32_t),
                                         &op_data->temp_buffer_idx);
    op_data->input_zp = input->params.zero_point;
    op_data->input_scale = input->params.scale;
    op_data->output_zp = output->params.zero_point;
    op_data->output_scale = output->params.scale;
  }

  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node));
  // TODO(b/144955155): Support uint8_t(b/144955155) and int8_t(b/144955018)
  return kTfLiteOk;
}

void ResolveAxis(const int* axis_data, int axis_count,
                 tflite::MeanParams* op_params) {
  int i = 0;
  for (; i < axis_count; ++i) {
    op_params->axis[i] = static_cast<int16_t>(axis_data[i]);
  }
  for (; i < 4; ++i) {
    op_params->axis[i] = 1;
  }
  op_params->axis_count = axis_count;
}

}  // namespace tflite
