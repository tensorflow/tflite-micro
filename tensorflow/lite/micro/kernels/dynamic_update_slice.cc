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
#include "tensorflow/lite/micro/kernels/dynamic_update_slice.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

constexpr int kMaxDimensions = RuntimeShape::kMaxSmallSize;

namespace {

void CalculateClampedStartIndices(int num_dims, const int64_t* raw_indices_data,
                                  const int32_t* input_dims_data,
                                  const int32_t* update_dims_data,
                                  int32_t* clamped_start_indices_output) {
  for (int i = 0; i < num_dims; ++i) {
    clamped_start_indices_output[i] = static_cast<int32_t>(
        std::min<int64_t>(std::max<int64_t>(0, raw_indices_data[i]),
                          input_dims_data[i] - update_dims_data[i]));
  }
  return;
}

// Recursive helper for N-dimensional slice update.
template <typename T>
void UpdateSliceRecursive(int current_dim, int max_dims,
                          const int32_t* output_strides,
                          const int32_t* update_strides,
                          const int32_t* update_dims_data,
                          const T* update_tensor_data,
                          const int32_t* clamped_start_indices,
                          T* output_tensor_data) {
  if (current_dim == max_dims) return;
  output_tensor_data +=
      clamped_start_indices[current_dim] * output_strides[current_dim];
  if (current_dim == max_dims - 1) {
    std::memcpy(output_tensor_data, update_tensor_data,
                update_dims_data[max_dims - 1] * sizeof(T));
  } else {
    for (int i = 0; i < update_dims_data[current_dim]; ++i) {
      UpdateSliceRecursive<T>(current_dim + 1, max_dims, output_strides,
                              update_strides, update_dims_data,
                              update_tensor_data, clamped_start_indices,
                              output_tensor_data);
      output_tensor_data += output_strides[current_dim];
      update_tensor_data += update_strides[current_dim];
    }
  }
}

// Main dispatch function for Eval, templated on data type.
template <typename T>
void EvalImpl(const TfLiteEvalTensor* operand_eval,
              const TfLiteEvalTensor* update_eval, const int64_t* indices_eval,
              TfLiteEvalTensor* output_eval) {
  const RuntimeShape operand_shape =
      tflite::micro::GetTensorShape(operand_eval);
  const RuntimeShape update_shape = tflite::micro::GetTensorShape(update_eval);
  const T* update_tensor_data = tflite::micro::GetTensorData<T>(update_eval);
  T* output_tensor_data = tflite::micro::GetTensorData<T>(output_eval);

  const int num_dims = operand_shape.DimensionsCount();
  if (operand_shape.FlatSize() == update_shape.FlatSize()) {
    std::memcpy(output_tensor_data, update_tensor_data,
                ElementCount(*operand_eval->dims) * sizeof(T));
    return;
  }

  // If the operation is not done in-place, copy the input data to the output.
  if (operand_eval->data.data != output_eval->data.data) {
    std::memcpy(output_eval->data.data, operand_eval->data.data,
                ElementCount(*operand_eval->dims) * sizeof(T));
  }

  // If update tensor is empty, no actual update is needed after operand copy.
  if (ElementCount(*update_eval->dims) == 0) {
    return;
  }

  // Calculate clamped start indices (stack-allocated)
  int32_t clamped_start_indices[kMaxDimensions];
  CalculateClampedStartIndices(num_dims, indices_eval, operand_shape.DimsData(),
                               update_shape.DimsData(), clamped_start_indices);

  // Calculate strides (stack-allocated)
  int32_t output_stride[kMaxDimensions];
  int32_t update_stride[kMaxDimensions];
  output_stride[num_dims - 1] = 1;
  update_stride[num_dims - 1] = 1;
  for (int i = num_dims - 2; i >= 0; --i) {
    output_stride[i] = output_stride[i + 1] * operand_shape.Dims(i + 1);
    update_stride[i] = update_stride[i + 1] * update_shape.Dims(i + 1);
  }

  // Perform the N-dimensional update
  // The recursive function needs base pointers and initial offsets.
  UpdateSliceRecursive<T>(
      /*current_dim=*/0, num_dims, output_stride, update_stride,
      update_shape.DimsData(), update_tensor_data, clamped_start_indices,
      output_tensor_data);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Use MicroContext to allocate temporary tensors for inspection
  // This is a robust pattern shown in EMBEDDING_LOOKUP.
  TfLiteTensor* operand = micro_context->AllocateTempInputTensor(
      node, kDynamicUpdateSliceOperandTensor);
  TF_LITE_ENSURE(context, operand != nullptr);

  TfLiteTensor* update = micro_context->AllocateTempInputTensor(
      node, kDynamicUpdateSliceUpdateTensor);
  TF_LITE_ENSURE(context, update != nullptr);

  TfLiteTensor* start_indices = micro_context->AllocateTempInputTensor(
      node, kDynamicUpdateSliceStartIndicesTensor);
  TF_LITE_ENSURE(context, start_indices != nullptr);

  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kDynamicUpdateSliceOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  // Type checks
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, update->type);
  TF_LITE_ENSURE(context, start_indices->type == kTfLiteInt32 ||
                              start_indices->type == kTfLiteInt64);

  TF_LITE_ENSURE_EQ(context, NumDimensions(start_indices), 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(start_indices, 0),
                    NumDimensions(operand));

  TF_LITE_ENSURE_EQ(context, NumDimensions(update), NumDimensions(operand));
  // Check that update dimensions are not larger than operand dimensions
  for (int i = 0; i < NumDimensions(operand); ++i) {
    TF_LITE_ENSURE(context,
                   SizeOfDimension(update, i) <= SizeOfDimension(operand, i));
  }

  // Deallocate temporary tensors
  micro_context->DeallocateTempTfLiteTensor(operand);
  micro_context->DeallocateTempTfLiteTensor(update);
  micro_context->DeallocateTempTfLiteTensor(start_indices);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* operand_eval = tflite::micro::GetEvalInput(
      context, node, kDynamicUpdateSliceOperandTensor);
  const TfLiteEvalTensor* update_eval = tflite::micro::GetEvalInput(
      context, node, kDynamicUpdateSliceUpdateTensor);
  const TfLiteEvalTensor* indices_eval = tflite::micro::GetEvalInput(
      context, node, kDynamicUpdateSliceStartIndicesTensor);
  TfLiteEvalTensor* output_eval = tflite::micro::GetEvalOutput(
      context, node, kDynamicUpdateSliceOutputTensor);

  const auto& input_shape = tflite::micro::GetTensorShape(operand_eval);
  const int input_dims = input_shape.DimensionsCount();
  int64_t indices_data_i64[kMaxDimensions];
  if (indices_eval->type == kTfLiteInt32) {
    for (int i = 0; i < input_dims; i++)
      indices_data_i64[i] = static_cast<int64_t>(indices_eval->data.i32[i]);
  } else if (indices_eval->type == kTfLiteInt64) {
    for (int i = 0; i < input_dims; i++)
      indices_data_i64[i] = indices_eval->data.i64[i];
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "DynamicUpdateSlice only currently supports "
                       "int32 or int64 indices type, got %d.",
                       indices_eval->type);
    return kTfLiteError;
  }
  // Dispatch based on tensor type
  switch (operand_eval->type) {
    case kTfLiteFloat32:
      EvalImpl<float>(operand_eval, update_eval, indices_data_i64, output_eval);
      break;
    case kTfLiteInt8:
      EvalImpl<int8_t>(operand_eval, update_eval, indices_data_i64,
                       output_eval);
      break;
    case kTfLiteInt16:
      EvalImpl<int16_t>(operand_eval, update_eval, indices_data_i64,
                        output_eval);
      break;
    case kTfLiteInt32:
      EvalImpl<int32_t>(operand_eval, update_eval, indices_data_i64,
                        output_eval);
      break;
    default:
      MicroPrintf("DYNAMIC_UPDATE_SLICE: Operand type %s not supported.",
                  TfLiteTypeGetName(operand_eval->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_DYNAMIC_UPDATE_SLICE() {
  return tflite::micro::RegisterOp(/*init=*/nullptr, /*prepare=*/Prepare,
                                   /*invoke=*/Eval);
}

}  // namespace tflite
