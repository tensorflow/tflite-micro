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

#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace elementwise {
namespace {

const char kAbsName[] = "Abs";
const char kRsqrtName[] = "Rsqrt";

struct OpData {
  int32_t multiplier;
  int shift;
  int input_offset;
  int output_offset;
  bool needs_rescale;
};

bool IsNumericSupportedType(const TfLiteType type) {
  return type == kTfLiteFloat32;
}

bool IsLogicalSupportedType(const TfLiteType type) {
  return type == kTfLiteBool;
}

bool IsRsqrtSupportedType(const TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteInt8;
}

inline void SetRsqrtOutputMultiplier(const float input_scale,
                                     const float output_scale,
                                     int32_t* multiplier, int* shift) {
  const double scale = 1.0 / (std::sqrt(static_cast<double>(input_scale)) *
                              static_cast<double>(output_scale));
  QuantizeMultiplier(scale, multiplier, shift);
}

typedef bool (*IsSupportedType)(TfLiteType);
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node,
                            IsSupportedType is_supported_type,
                            const char* op_name) {
  MicroContext* micro_context = GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (!is_supported_type(input->type)) {
    TF_LITE_KERNEL_LOG(context, "Input data type %s (%d) is not supported.",
                       TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }

  // For int16 type input, we support both quantized and non-quantized
  // evaluation.
  if (input->type == kTfLiteInt8 ||
      (input->type == kTfLiteInt16 &&
       input->quantization.type != kTfLiteNoQuantization)) {
    auto* op_data = static_cast<OpData*>(node->user_data);
    TF_LITE_ENSURE_EQ(context, input->quantization.type,
                      kTfLiteAffineQuantization);
    TF_LITE_ENSURE_EQ(context, output->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* input_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    const auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);
    TF_LITE_ENSURE(context, input_params != nullptr);
    TF_LITE_ENSURE(context, input_params->scale != nullptr);
    TF_LITE_ENSURE(context, input_params->scale->size > 0);
    TF_LITE_ENSURE(context, input_params->zero_point->size > 0);
    TF_LITE_ENSURE(context, output_params != nullptr);
    TF_LITE_ENSURE(context, output_params->scale != nullptr);
    TF_LITE_ENSURE(context, output_params->scale->size > 0);
    TF_LITE_ENSURE(context, output_params->zero_point->size > 0);
    op_data->input_offset = input_params->zero_point->data[0];
    op_data->output_offset = output_params->zero_point->data[0];
    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, op_data->input_offset, 0);
      TF_LITE_ENSURE_EQ(context, op_data->output_offset, 0);
    }
    const float input_scale = input_params->scale->data[0];
    const float output_scale = output_params->scale->data[0];
    op_data->needs_rescale = input_scale != output_scale;
    if (op_name == kRsqrtName) {
      SetRsqrtOutputMultiplier(input_scale, output_scale, &op_data->multiplier,
                               &op_data->shift);
    }
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

template <typename T>
inline TfLiteStatus EvalImpl(
    TfLiteContext* context, TfLiteNode* node, std::function<T(T)> func,
    std::function<TfLiteStatus(TfLiteContext*, T)> validate_input_func,
    TfLiteType expected_type) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, expected_type);
  const size_t num_elements = ElementCount(*input->dims);
  const T* in_data = tflite::micro::GetTensorData<T>(input);
  T* out_data = tflite::micro::GetTensorData<T>(output);
  for (size_t i = 0; i < num_elements; ++i) {
    if (validate_input_func) {
      TF_LITE_ENSURE_OK(context, validate_input_func(context, in_data[i]));
    }
    out_data[i] = func(in_data[i]);
  }
  return kTfLiteOk;
}

template <typename T>
inline TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                             T func(T), TfLiteType expected_type) {
  return EvalImpl<T>(context, node, func, /*validate_input_func=*/nullptr,
                     expected_type);
}

inline TfLiteStatus EvalNumeric(TfLiteContext* context, TfLiteNode* node,
                                float float_func(float)) {
  return EvalImpl<float>(context, node, float_func, kTfLiteFloat32);
}

inline TfLiteStatus EvalLogical(TfLiteContext* context, TfLiteNode* node,
                                bool bool_func(bool)) {
  return EvalImpl<bool>(context, node, bool_func, kTfLiteBool);
}

void* ElementWiseQuantizedInit(TfLiteContext* context, const char* buffer,
                               size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus AbsEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::abs);
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::sin);
}

TfLiteStatus CosEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::cos);
}

TfLiteStatus LogEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::log);
}

TfLiteStatus SqrtEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::sqrt);
}

TfLiteStatus RsqrtEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                                TfLiteType type) {
  const auto* op_data = static_cast<const OpData*>(node->user_data);
  const int32_t kMin = std::numeric_limits<int8_t>::min();
  const int32_t kMax = std::numeric_limits<int8_t>::max();
  std::function<TfLiteStatus(TfLiteContext * context_func, int8_t)>
      validate_input_func = [&](TfLiteContext* context_func, int8_t i) {
        TF_LITE_ENSURE_MSG(context, i >= op_data->input_offset,
                           "Rsqrt is only defined for positive values");
        return kTfLiteOk;
      };

  std::function<int8_t(int8_t)> func = [&](int8_t i) {
    const int32_t value = (i - op_data->input_offset);
    const int32_t kShift = 20;  // Shift to keep value integer.
    if (value == 0) {
      // Assume that any value close to 0 represents the max output value.
      return static_cast<int8_t>(kMax);
    }
    int32_t inv_sqrt_multiplier;
    int inv_sqrt_shift;
    GetInvSqrtQuantizedMultiplierExp(value, kReverseShift, &inv_sqrt_multiplier,
                                     &inv_sqrt_shift);
    const int32_t data = MultiplyByQuantizedMultiplier(
        static_cast<int32_t>(1), inv_sqrt_multiplier, inv_sqrt_shift + kShift);
    const int32_t output =
        MultiplyByQuantizedMultiplier(data, op_data->multiplier,
                                      op_data->shift - kShift) +
        op_data->output_offset;
    return static_cast<int8_t>(std::min(std::max(output, kMin), kMax));
  };

  return EvalImpl<int8_t>(context, node, func, validate_input_func, type);
}

TfLiteStatus RsqrtEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32:
      return EvalImpl<float>(
          context, node, [](float f) { return 1.f / std::sqrt(f); },
          input->type);
    case kTfLiteInt8:
      return RsqrtEvalQuantized(context, node, input->type);
    default:
      MicroPrintf("Current data type %s is not supported.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus SquareEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, [](float f) { return f * f; });
}

TfLiteStatus LogicalNotEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalLogical(context, node, [](bool v) { return !v; });
}

}  // namespace
}  // namespace elementwise

// Given a function...
// template<int T>
// int Foo(int b)
//
// typedef int(*Bar)(int);
//
// MSVC2015 will not see Foo<10> as the same type as Bar.
//
// This works around the issue by instantiating wrapper methods around
// elementwise::GenericPrepare() rather than using a templated
// elementwise::GenericPrepare method.
#define GENERIC_PREPARE(function_name, is_supported_type_function, type_name)  \
  static TfLiteStatus function_name(TfLiteContext* context,                    \
                                    TfLiteNode* node) {                        \
    return elementwise::GenericPrepare(context, node,                          \
                                       is_supported_type_function, type_name); \
  }

GENERIC_PREPARE(PrepareAbs, elementwise::IsNumericSupportedType,
                elementwise::kAbsName)

TfLiteRegistration Register_ABS() {
  return tflite::micro::RegisterOp(
      /*init=*/nullptr, PrepareAbs, elementwise::AbsEval);
}

GENERIC_PREPARE(PrepareSin, elementwise::IsNumericSupportedType, "Sin")

TfLiteRegistration Register_SIN() {
  return tflite::micro::RegisterOp(
      /*init=*/nullptr, PrepareSin, elementwise::SinEval);
}

GENERIC_PREPARE(PrepareCos, elementwise::IsNumericSupportedType, "Cos")

TfLiteRegistration Register_COS() {
  return tflite::micro::RegisterOp(
      /*init=*/nullptr, PrepareCos, elementwise::CosEval);
}

GENERIC_PREPARE(PrepareLog, elementwise::IsNumericSupportedType, "Log")

TfLiteRegistration Register_LOG() {
  return tflite::micro::RegisterOp(
      /*init=*/nullptr, PrepareLog, elementwise::LogEval);
}

GENERIC_PREPARE(PrepareSqrt, elementwise::IsNumericSupportedType, "Sqrt")

TfLiteRegistration Register_SQRT() {
  return tflite::micro::RegisterOp(
      /*init=*/nullptr, PrepareSqrt, elementwise::SqrtEval);
}

GENERIC_PREPARE(PrepareRsqrt, elementwise::IsRsqrtSupportedType,
                elementwise::kRsqrtName)

TfLiteRegistration Register_RSQRT() {
  return tflite::micro::RegisterOp(elementwise::ElementWiseQuantizedInit,
                                   PrepareRsqrt, elementwise::RsqrtEval);
}

GENERIC_PREPARE(PrepareSquare, elementwise::IsNumericSupportedType, "Square")

TfLiteRegistration Register_SQUARE() {
  return tflite::micro::RegisterOp(
      /*init=*/nullptr, PrepareSquare, elementwise::SquareEval);
}

GENERIC_PREPARE(PrepareNot, elementwise::IsLogicalSupportedType, "Not")

TfLiteRegistration Register_LOGICAL_NOT() {
  return tflite::micro::RegisterOp(
      /*init=*/nullptr, PrepareNot, elementwise::LogicalNotEval);
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
