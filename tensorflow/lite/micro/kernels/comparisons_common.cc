/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/kernels/comparisons.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/comparisons.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

template <typename T, template <typename> class OpFn>
void EvalComparisonNoScaling(const ComparisonParams& params,
                             const RuntimeShape& input1_shape,
                             const TfLiteEvalTensor* input1,
                             const RuntimeShape& input2_shape,
                             const TfLiteEvalTensor* input2,
                             const RuntimeShape& output_shape,
                             bool* output_data, bool requires_broadcast) {
  if (requires_broadcast) {
    reference_ops::BroadcastComparison4DSlowImpl<T, OpFn<T>>(
        params, input1_shape, tflite::micro::GetTensorData<T>(input1),
        input2_shape, tflite::micro::GetTensorData<T>(input2), output_shape,
        output_data);
  } else {
    reference_ops::ComparisonImpl<T, OpFn<T>>(
        params, input1_shape, tflite::micro::GetTensorData<T>(input1),
        input2_shape, tflite::micro::GetTensorData<T>(input2), output_shape,
        output_data);
  }
}

template <typename T, template <typename> class OpFn>
void EvalComparisonWithScaling(const ComparisonParams& params,
                               const RuntimeShape& input1_shape,
                               const TfLiteEvalTensor* input1,
                               const RuntimeShape& input2_shape,
                               const TfLiteEvalTensor* input2,
                               const RuntimeShape& output_shape,
                               bool* output_data, bool requires_broadcast) {
  if (requires_broadcast) {
    reference_ops::BroadcastComparison4DSlowWithScaling<T, OpFn<int32_t>>(
        params, input1_shape, tflite::micro::GetTensorData<T>(input1),
        input2_shape, tflite::micro::GetTensorData<T>(input2), output_shape,
        output_data);
  } else {
    reference_ops::ComparisonWithScaling<T, OpFn<int32_t>>(
        params, input1_shape, tflite::micro::GetTensorData<T>(input1),
        input2_shape, tflite::micro::GetTensorData<T>(input2), output_shape,
        output_data);
  }
}

template <template <typename> class OpFn, bool kAllowBool,
          bool kAllowInt16Quantized>
TfLiteStatus EvalComparisonReference(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto* data = static_cast<const OpDataComparisons*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kComparisonsInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kComparisonsOutputTensor);

  const RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  const RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  const RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  const bool requires_broadcast =
      !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteBool:
      if constexpr (kAllowBool) {
        EvalComparisonNoScaling<bool, OpFn>(data->params, input1_shape, input1,
                                            input2_shape, input2, output_shape,
                                            output_data, requires_broadcast);
        return kTfLiteOk;
      }
      break;
    case kTfLiteFloat32:
      EvalComparisonNoScaling<float, OpFn>(data->params, input1_shape, input1,
                                           input2_shape, input2, output_shape,
                                           output_data, requires_broadcast);
      return kTfLiteOk;
    case kTfLiteInt32:
      EvalComparisonNoScaling<int32_t, OpFn>(data->params, input1_shape, input1,
                                             input2_shape, input2, output_shape,
                                             output_data, requires_broadcast);
      return kTfLiteOk;
    case kTfLiteInt64:
      EvalComparisonNoScaling<int64_t, OpFn>(data->params, input1_shape, input1,
                                             input2_shape, input2, output_shape,
                                             output_data, requires_broadcast);
      return kTfLiteOk;
    case kTfLiteInt8:
      EvalComparisonWithScaling<int8_t, OpFn>(
          data->params, input1_shape, input1, input2_shape, input2,
          output_shape, output_data, requires_broadcast);
      return kTfLiteOk;
    case kTfLiteInt16:
      if (!data->is_quantized) {
        EvalComparisonNoScaling<int16_t, OpFn>(
            data->params, input1_shape, input1, input2_shape, input2,
            output_shape, output_data, requires_broadcast);
        return kTfLiteOk;
      }
      if constexpr (kAllowInt16Quantized) {
        EvalComparisonWithScaling<int16_t, OpFn>(
            data->params, input1_shape, input1, input2_shape, input2,
            output_shape, output_data, requires_broadcast);
        return kTfLiteOk;
      }
      break;
    default:
      break;
  }

  MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input1->type),
              input1->type);
  return kTfLiteError;
}

}  // namespace

void* ComparisonsInit(TfLiteContext* context, const char* buffer,
                      size_t length) {
  void* data =
      context->AllocatePersistentBuffer(context, sizeof(OpDataComparisons));
  return new (data) OpDataComparisons();
}

TfLiteStatus ComparisonsPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  auto* data = static_cast<OpDataComparisons*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input1 =
      micro_context->AllocateTempInputTensor(node, kComparisonsInputTensor1);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* input2 =
      micro_context->AllocateTempInputTensor(node, kComparisonsInputTensor2);
  TF_LITE_ENSURE(context, input2 != nullptr);

  data->is_quantized = (input1->type == kTfLiteInt8) ||
                       (input1->quantization.type != kTfLiteNoQuantization);

  if (input1->type == kTfLiteInt8 ||
      (input1->type == kTfLiteInt16 && data->is_quantized)) {
    const int32_t input1_offset = -input1->params.zero_point;
    const int32_t input2_offset = -input2->params.zero_point;
    constexpr int kLeftShift = 8;

    int32_t input1_multiplier;
    int input1_shift;
    QuantizeMultiplier(static_cast<double>(input1->params.scale),
                       &input1_multiplier, &input1_shift);
    int32_t input2_multiplier;
    int input2_shift;
    QuantizeMultiplier(static_cast<double>(input2->params.scale),
                       &input2_multiplier, &input2_shift);

    data->params.left_shift = kLeftShift;
    data->params.input1_offset = input1_offset;
    data->params.input1_multiplier = input1_multiplier;
    data->params.input1_shift = input1_shift;
    data->params.input2_offset = input2_offset;
    data->params.input2_multiplier = input2_multiplier;
    data->params.input2_shift = input2_shift;
  }

  micro_context->DeallocateTempTfLiteTensor(input1);
  micro_context->DeallocateTempTfLiteTensor(input2);

  return kTfLiteOk;
}

TfLiteStatus EqualEvalReference(TfLiteContext* context, TfLiteNode* node) {
  return EvalComparisonReference<reference_ops::EqualFn, /*kAllowBool=*/true,
                                 /*kAllowInt16Quantized=*/true>(context, node);
}

TfLiteStatus NotEqualEvalReference(TfLiteContext* context, TfLiteNode* node) {
  return EvalComparisonReference<reference_ops::NotEqualFn,
                                 /*kAllowBool=*/true,
                                 /*kAllowInt16Quantized=*/true>(context, node);
}

TfLiteStatus GreaterEvalReference(TfLiteContext* context, TfLiteNode* node) {
  return EvalComparisonReference<reference_ops::GreaterFn,
                                 /*kAllowBool=*/false,
                                 /*kAllowInt16Quantized=*/false>(context, node);
}

TfLiteStatus GreaterEqualEvalReference(TfLiteContext* context,
                                       TfLiteNode* node) {
  return EvalComparisonReference<reference_ops::GreaterEqualFn,
                                 /*kAllowBool=*/false,
                                 /*kAllowInt16Quantized=*/false>(context, node);
}

TfLiteStatus LessEvalReference(TfLiteContext* context, TfLiteNode* node) {
  return EvalComparisonReference<reference_ops::LessFn, /*kAllowBool=*/false,
                                 /*kAllowInt16Quantized=*/false>(context, node);
}

TfLiteStatus LessEqualEvalReference(TfLiteContext* context, TfLiteNode* node) {
  return EvalComparisonReference<reference_ops::LessEqualFn,
                                 /*kAllowBool=*/false,
                                 /*kAllowInt16Quantized=*/false>(context, node);
}

}  // namespace micro
}  // namespace tflite
