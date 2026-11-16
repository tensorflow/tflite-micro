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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/activations.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace micro {

template <typename T>
void CalculateReluOpData(const TfLiteTensor* input, TfLiteTensor* output,
                         ReluOpData* data) {
  float act_min = 0.0;
  float act_max = std::numeric_limits<float>::infinity();
  double real_multiplier = static_cast<double>(input->params.scale) /
                           static_cast<double>(output->params.scale);

  QuantizeMultiplier(real_multiplier, &data->params.output_multiplier,
                     &data->params.output_shift);

  data->params.quantized_activation_min = std::max(
      static_cast<int32_t>(std::numeric_limits<T>::min()),
      output->params.zero_point +
          static_cast<int32_t>(roundf(act_min / output->params.scale)));
  data->params.quantized_activation_max =
      act_max == std::numeric_limits<float>::infinity()
          ? static_cast<int32_t>(std::numeric_limits<T>::max())
          : std::min(static_cast<int32_t>(std::numeric_limits<T>::max()),
                     output->params.zero_point +
                         static_cast<int32_t>(
                             roundf(act_max / output->params.scale)));
  data->params.input_offset = input->params.zero_point;
  data->params.output_offset = output->params.zero_point;
}

void ReluFloat(const RuntimeShape& input_shape, const float* input_data,
               const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float lower = 0.0f;
    const float clamped = val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

void Relu6Float(const RuntimeShape& input_shape, const float* input_data,
                const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float upper = 6.0f;
    const float lower = 0.0f;
    const float clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

void* ReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  void* data = context->AllocatePersistentBuffer(context, sizeof(ReluOpData));
  return new (data) ReluOpData();
}

TfLiteStatus ReluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  ReluOpData* data = static_cast<ReluOpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kActivationsInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kActivationsOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(context, HaveSameShapes(input, output));

  if (input->type == kTfLiteInt8) {
    CalculateReluOpData<int8_t>(input, output, data);
  } else if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    CalculateReluOpData<int16_t>(input, output, data);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

template <>
void ReluEvalReference<float>(const ReluOpData& data,
                              const TfLiteEvalTensor* input,
                              TfLiteEvalTensor* output) {
  ReluFloat(tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<float>(input),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<float>(output));
}

template <>
void ReluEvalReference<int8_t>(const ReluOpData& data,
                               const TfLiteEvalTensor* input,
                               TfLiteEvalTensor* output) {
  ReluQuantized<int8_t>(data, tflite::micro::GetTensorShape(input),
                        tflite::micro::GetTensorShape(output),
                        tflite::micro::GetTensorData<int8_t>(input),
                        tflite::micro::GetTensorData<int8_t>(output));
}

template <>
void ReluEvalReference<int16_t>(const ReluOpData& data,
                                const TfLiteEvalTensor* input,
                                TfLiteEvalTensor* output) {
  ReluQuantized<int16_t>(data, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<int16_t>(input),
                         tflite::micro::GetTensorData<int16_t>(output));
}

void* Relu6Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* data = context->AllocatePersistentBuffer(context, sizeof(Relu6OpData));
  return new (data) Relu6OpData();
}

TfLiteStatus Relu6Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  Relu6OpData* data = static_cast<Relu6OpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kActivationsInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kActivationsOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(context, HaveSameShapes(input, output));

  if (input->type == kTfLiteInt8) {
    data->zero = input->params.zero_point;
    data->six = FloatToQuantizedType<int8_t>(6.0f, input->params.scale,
                                             input->params.zero_point);
    TF_LITE_ENSURE(context, data->zero >= INT8_MIN && data->zero <= INT8_MAX);
    TF_LITE_ENSURE(context, data->six >= INT8_MIN && data->six <= INT8_MAX);
  } else if (input->type == kTfLiteInt16) {
    data->zero = input->params.zero_point;
    data->six = FloatToQuantizedType<int16_t>(6.0f, input->params.scale,
                                              input->params.zero_point);
    TF_LITE_ENSURE(context, data->zero >= INT16_MIN && data->zero <= INT16_MAX);
    TF_LITE_ENSURE(context, data->six >= INT16_MIN && data->six <= INT16_MAX);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

template <>
void Relu6EvalReference<float>(const Relu6OpData& data,
                               const TfLiteEvalTensor* input,
                               TfLiteEvalTensor* output) {
  Relu6Float(tflite::micro::GetTensorShape(input),
             tflite::micro::GetTensorData<float>(input),
             tflite::micro::GetTensorShape(output),
             tflite::micro::GetTensorData<float>(output));
}

template <>
void Relu6EvalReference<int8_t>(const Relu6OpData& data,
                                const TfLiteEvalTensor* input,
                                TfLiteEvalTensor* output) {
  Relu6Quantized<int8_t>(data.zero, data.six,
                         tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<int8_t>(input),
                         tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<int8_t>(output));
}

template <>
void Relu6EvalReference<int16_t>(const Relu6OpData& data,
                                 const TfLiteEvalTensor* input,
                                 TfLiteEvalTensor* output) {
  Relu6Quantized<int16_t>(data.zero, data.six,
                          tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<int16_t>(input),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<int16_t>(output));
}

}  // namespace micro
}  // namespace tflite
