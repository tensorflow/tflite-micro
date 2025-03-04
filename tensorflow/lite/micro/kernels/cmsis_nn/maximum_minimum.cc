/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/maximum_minimum.h"

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

cmsis_nn_dims FillVariableShape(int32_t rank, int32_t* tensor_dims) {
  if (rank == 4) {
    return {tensor_dims[0], tensor_dims[1], tensor_dims[2], tensor_dims[3]};
  } else if (rank == 3) {
    return {1, tensor_dims[0], tensor_dims[1], tensor_dims[2]};
  } else if (rank == 2) {
    return {1, 1, tensor_dims[0], tensor_dims[1]};
  } else {
    return {1, 1, 1, 1};
  }
}

TfLiteStatus EvalMaximum(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  RuntimeShape input_1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input_2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

  cmsis_nn_dims input_1_dims = FillVariableShape(
      input_1_shape.DimensionsCount(), input_1_shape.DimsData());
  cmsis_nn_dims input_2_dims = FillVariableShape(
      input_2_shape.DimensionsCount(), input_2_shape.DimsData());
  cmsis_nn_dims output_dims = FillVariableShape(output_shape.DimensionsCount(),
                                                output_shape.DimsData());

  switch (op_context.output->type) {
    case kTfLiteInt8:
      cmsis_nn_context ctx;
      ctx.buf = nullptr;
      ctx.size = 0;

      arm_maximum_s8(
          &ctx, tflite::micro::GetTensorData<int8_t>(input1), &input_1_dims,
          tflite::micro::GetTensorData<int8_t>(input2), &input_2_dims,
          tflite::micro::GetTensorData<int8_t>(output), &output_dims);
      break;
    case kTfLiteFloat32:
      TFLiteOperation<float, MaximumOp>(context, node, op_context);
      break;
    case kTfLiteInt16:
      TFLiteOperation<int16_t, MaximumOp>(context, node, op_context);
      break;
    case kTfLiteInt32:
      TFLiteOperation<int32_t, MaximumOp>(context, node, op_context);
      break;
    case kTfLiteInt64:
      TFLiteOperation<int64_t, MaximumOp>(context, node, op_context);
      break;
    default:
      MicroPrintf("Type %s (%d) is not supported by Maximum/Minimum.",
                  TfLiteTypeGetName(op_context.output->type),
                  op_context.output->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMaximumInt8(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  RuntimeShape input_1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input_2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

  cmsis_nn_dims input_1_dims = FillVariableShape(
      input_1_shape.DimensionsCount(), input_1_shape.DimsData());
  cmsis_nn_dims input_2_dims = FillVariableShape(
      input_2_shape.DimensionsCount(), input_2_shape.DimsData());
  cmsis_nn_dims output_dims = FillVariableShape(output_shape.DimensionsCount(),
                                                output_shape.DimsData());

  switch (op_context.output->type) {
    case kTfLiteInt8:
      cmsis_nn_context ctx;
      ctx.buf = nullptr;
      ctx.size = 0;

      arm_maximum_s8(
          &ctx, tflite::micro::GetTensorData<int8_t>(input1), &input_1_dims,
          tflite::micro::GetTensorData<int8_t>(input2), &input_2_dims,
          tflite::micro::GetTensorData<int8_t>(output), &output_dims);
      break;
    default:
      MicroPrintf("Type %s (%d) is not supported by Maximum Int8 Registration.",
                  TfLiteTypeGetName(op_context.output->type),
                  op_context.output->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMinimum(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  RuntimeShape input_1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input_2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

  cmsis_nn_dims input_1_dims = FillVariableShape(
      input_1_shape.DimensionsCount(), input_1_shape.DimsData());
  cmsis_nn_dims input_2_dims = FillVariableShape(
      input_2_shape.DimensionsCount(), input_2_shape.DimsData());
  cmsis_nn_dims output_dims = FillVariableShape(output_shape.DimensionsCount(),
                                                output_shape.DimsData());

  switch (op_context.output->type) {
    case kTfLiteInt8:
      cmsis_nn_context ctx;
      ctx.buf = nullptr;
      ctx.size = 0;

      arm_minimum_s8(
          &ctx, tflite::micro::GetTensorData<int8_t>(input1), &input_1_dims,
          tflite::micro::GetTensorData<int8_t>(input2), &input_2_dims,
          tflite::micro::GetTensorData<int8_t>(output), &output_dims);
      break;
    case kTfLiteFloat32:
      TFLiteOperation<float, MinimumOp>(context, node, op_context);
      break;
    case kTfLiteInt16:
      TFLiteOperation<int16_t, MinimumOp>(context, node, op_context);
      break;
    case kTfLiteInt32:
      TFLiteOperation<int32_t, MinimumOp>(context, node, op_context);
      break;
    case kTfLiteInt64:
      TFLiteOperation<int64_t, MinimumOp>(context, node, op_context);
      break;
    default:
      MicroPrintf("Type %s (%d) is not supported by Maximum/Minimum.",
                  TfLiteTypeGetName(op_context.output->type),
                  op_context.output->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMinimumInt8(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  RuntimeShape input_1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input_2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

  cmsis_nn_dims input_1_dims = FillVariableShape(
      input_1_shape.DimensionsCount(), input_1_shape.DimsData());
  cmsis_nn_dims input_2_dims = FillVariableShape(
      input_2_shape.DimensionsCount(), input_2_shape.DimsData());
  cmsis_nn_dims output_dims = FillVariableShape(output_shape.DimensionsCount(),
                                                output_shape.DimsData());

  switch (op_context.output->type) {
    case kTfLiteInt8:
      cmsis_nn_context ctx;
      ctx.buf = nullptr;
      ctx.size = 0;

      arm_minimum_s8(
          &ctx, tflite::micro::GetTensorData<int8_t>(input1), &input_1_dims,
          tflite::micro::GetTensorData<int8_t>(input2), &input_2_dims,
          tflite::micro::GetTensorData<int8_t>(output), &output_dims);
      break;
    default:
      MicroPrintf("Type %s (%d) is not supported by Minimum Int8 registration.",
                  TfLiteTypeGetName(op_context.output->type),
                  op_context.output->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_MAXIMUM() {
  return tflite::micro::RegisterOp(nullptr, nullptr, EvalMaximum);
}

TFLMRegistration Register_MINIMUM() {
  return tflite::micro::RegisterOp(nullptr, nullptr, EvalMinimum);
}

TFLMRegistration Register_MAXIMUM_INT8() {
  return tflite::micro::RegisterOp(nullptr, nullptr, EvalMaximumInt8);
}

TFLMRegistration Register_MINIMUM_INT8() {
  return tflite::micro::RegisterOp(nullptr, nullptr, EvalMinimumInt8);
}

}  // namespace tflite
