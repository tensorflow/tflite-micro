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

#include <cstddef>
#include <cstdint>
#include <new>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/prelu.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/prelu.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

TfLiteStatus CalculatePreluParams(const TfLiteTensor* input,
                                  const TfLiteTensor* alpha,
                                  TfLiteTensor* output, PreluParams* params) {
  if (output->type == kTfLiteInt8 || output->type == kTfLiteInt16) {
    double real_multiplier_1 = static_cast<double>(input->params.scale) /
                               static_cast<double>(output->params.scale);
    double real_multiplier_2 = static_cast<double>(input->params.scale) *
                               static_cast<double>(alpha->params.scale) /
                               static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier_1, &params->output_multiplier_1,
                       &params->output_shift_1);
    QuantizeMultiplier(real_multiplier_2, &params->output_multiplier_2,
                       &params->output_shift_2);

    params->input_offset = -input->params.zero_point;
    params->alpha_offset = -alpha->params.zero_point;
    params->output_offset = output->params.zero_point;
  }

  return kTfLiteOk;
}

void BroadcastPrelu4DSlowFloat(const RuntimeShape& unextended_input1_shape,
                               const float* input1_data,
                               const RuntimeShape& unextended_input2_shape,
                               const float* input2_data,
                               const RuntimeShape& unextended_output_shape,
                               float* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          auto out_idx = Offset(output_shape, b, y, x, c);
          auto in1_idx = SubscriptToIndex(desc1, b, y, x, c);
          auto in2_idx = SubscriptToIndex(desc2, b, y, x, c);
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = in1_val >= 0.0f ? in1_val : in1_val * in2_val;
        }
      }
    }
  }
}

void* PreluInitReference(TfLiteContext* context, const char* buffer,
                         size_t length) {
  void* data = context->AllocatePersistentBuffer(context, sizeof(PreluParams));
  return new (data) PreluParams();
}

TfLiteStatus PreparePreluParameters(TfLiteContext* context, TfLiteNode* node,
                                    PreluParams* params) {
  TF_LITE_ENSURE(context, params != nullptr);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kPreluInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* alpha =
      micro_context->AllocateTempInputTensor(node, kPreluAlphaTensor);
  TF_LITE_ENSURE(context, alpha != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kPreluOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (output->type == kTfLiteFloat32) {
    TF_LITE_ENSURE_TYPES_EQ(context, alpha->type, kTfLiteFloat32);
  } else if (output->type == kTfLiteInt8 || output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_TYPES_EQ(context, alpha->type, kTfLiteInt8);
  }

  TF_LITE_ENSURE_OK(context,
                    CalculatePreluParams(input, alpha, output, params));

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(alpha);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus PreluPrepareReference(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  PreluParams* params = static_cast<PreluParams*>(node->user_data);
  return PreparePreluParameters(context, node, params);
}

template <>
void PreluEvalReference<float>(const PreluParams& params,
                               const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* alpha,
                               TfLiteEvalTensor* output) {
  BroadcastPrelu4DSlowFloat(tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<float>(input),
                            tflite::micro::GetTensorShape(alpha),
                            tflite::micro::GetTensorData<float>(alpha),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<float>(output));
}

template <>
void PreluEvalReference<int8_t>(const PreluParams& params,
                                const TfLiteEvalTensor* input,
                                const TfLiteEvalTensor* alpha,
                                TfLiteEvalTensor* output) {
  reference_ops::BroadcastPrelu4DSlow(
      params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(alpha),
      tflite::micro::GetTensorData<int8_t>(alpha),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
}

template <>
void PreluEvalReference<int16_t>(const PreluParams& params,
                                 const TfLiteEvalTensor* input,
                                 const TfLiteEvalTensor* alpha,
                                 TfLiteEvalTensor* output) {
  reference_ops::BroadcastPrelu4DSlow(
      params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int16_t>(input),
      tflite::micro::GetTensorShape(alpha),
      tflite::micro::GetTensorData<int8_t>(alpha),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int16_t>(output));
}

}  // namespace micro
}  // namespace tflite
