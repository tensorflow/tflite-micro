/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/pad.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pad.h"

namespace tflite {

void* PadInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus PadPrepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE(context, NumInputs(node) == 2 || NumInputs(node) == 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, /*index=*/0);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* paddings =
      micro_context->AllocateTempInputTensor(node, /*index=*/1);
  TF_LITE_ENSURE(context, paddings != nullptr);
  TfLiteTensor* constant_values =
      NumInputs(node) == 3
          ? micro_context->AllocateTempInputTensor(node, /*index=*/2)
          : nullptr;
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, /*index=*/0);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  // Current implementations rely on the inputs being <= 4D.
  TF_LITE_ENSURE(context, NumDimensions(input) <=
                              reference_ops::PadKernelMaxDimensionCount());

  if (constant_values != nullptr) {
    TF_LITE_ENSURE_EQ(context, input->type, constant_values->type);
    // Ensure that constant_values is a scalar.
    TF_LITE_ENSURE_EQ(context, NumElements(constant_values), 1);
  }

  // There must be a pair of paddings for each output dimension.
  TF_LITE_ENSURE_EQ(context, GetTensorShape(paddings).FlatSize(),
                    output->dims->size * 2);

  // On Micro, outputs must be properly sized by the converter.
  // NOTE: This data is only available because the paddings buffer is stored in
  // the flatbuffer:
  TF_LITE_ENSURE_MSG(context, IsConstantTensor(paddings),
                     "Non-constant >paddings< tensor is not supported");
  const int32_t* paddings_data = GetTensorData<int32_t>(paddings);
  for (int i = 0; i < output->dims->size; i++) {
    int output_dim = output->dims->data[i];
    int expected_dim =
        input->dims->data[i] + paddings_data[i * 2] + paddings_data[i * 2 + 1];
    TF_LITE_ENSURE_EQ(context, output_dim, expected_dim);
  }

  // Calculate OpData:
  data->params.resizing_category = ResizingCategory::kGenericResize;
  const int paddings_total = GetTensorShape(paddings).FlatSize();
  if (paddings_total == 8 && (paddings_data[0] == 0 && paddings_data[1] == 0) &&
      (paddings_data[6] == 0 && paddings_data[7] == 0)) {
    data->params.resizing_category = ResizingCategory::kImageStyle;
  }

  const int num_input_dimensions = NumDimensions(input);
  data->params.left_padding_count = num_input_dimensions;
  data->params.right_padding_count = num_input_dimensions;

  for (int idx = num_input_dimensions - 1; idx >= 0; --idx) {
    data->params.left_padding[idx] = paddings_data[idx * 2];
    data->params.right_padding[idx] = paddings_data[idx * 2 + 1];
  }

  if (input->type == kTfLiteInt8) {
    if (constant_values == nullptr) {
      // Quantized Pad requires that 0 is represented in the quantized
      // range.
      TF_LITE_ENSURE(context, output->params.zero_point >=
                                  std::numeric_limits<int8_t>::min());
      TF_LITE_ENSURE(context, output->params.zero_point <=
                                  std::numeric_limits<int8_t>::max());
    } else {
      // Quantized Pad requires that 'constant_values' is represented in the
      // same quantized range as the input and output tensors.
      TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                        constant_values->params.zero_point);
      TF_LITE_ENSURE_EQ(context, static_cast<double>(output->params.scale),
                        static_cast<double>(constant_values->params.scale));
    }
    data->output_zero_point = output->params.zero_point;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(paddings);
  if (constant_values != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(constant_values);
  }
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

}  // namespace tflite
