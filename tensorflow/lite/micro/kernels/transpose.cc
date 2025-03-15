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
#include "tensorflow/lite/kernels/internal/reference/transpose.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kPermTensor = 1;
constexpr int kOutputTensor = 0;

struct TransposeContext {
  TransposeContext(TfLiteContext* context, TfLiteNode* node) {
    micro_context = GetMicroContext(context);
    input = micro_context->AllocateTempInputTensor(node, kInputTensor);
    perm = micro_context->AllocateTempInputTensor(node, kPermTensor);
    output = micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  }
  ~TransposeContext() {
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(perm);
    micro_context->DeallocateTempTfLiteTensor(output);
  }
  MicroContext* micro_context;
  TfLiteTensor* input;
  TfLiteTensor* perm;
  TfLiteTensor* output;
};

TfLiteStatus TransposePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TransposeContext op_context(context, node);

  // Ensure validity of input tensor.
  TF_LITE_ENSURE_MSG(context, NumDimensions(op_context.input) <= 5,
                     "Transpose op only supports 1D-5D input arrays.");
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                          op_context.output->type);

  int dims = NumDimensions(op_context.input);
  const int32_t* perm_data = GetTensorData<int32_t>(op_context.perm);

  // Ensure validity of the permutations tensor as a 1D tensor.
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.perm), 1);
  TF_LITE_ENSURE_EQ(context, op_context.perm->dims->data[0], dims);
  for (int idx = 0; idx < dims; ++idx) {
    TF_LITE_ENSURE_MSG(context, (perm_data[idx] >= 0 && perm_data[idx] < dims),
                       "Transpose op permutations array is out of bounds.");
  }

  return kTfLiteOk;
}

TfLiteStatus TransposeEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* perm_tensor =
      tflite::micro::GetEvalInput(context, node, kPermTensor);
  const int32_t* perm_data = perm_tensor->data.i32;
  const int size = perm_tensor->dims->data[0];
  TransposeParams params;
  params.perm_count = size;
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm_data[i];
  }

  // Transpose kernel only does rearranging values not numeric evaluations
  // on each cell. It's safe to implement per size of scalar type and this
  // trick keeps the total code size in a reasonable range.
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  switch (input->type) {
    case kTfLiteFloat32:
      reference_ops::Transpose(params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<float>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::Transpose(params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<int8_t>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<int8_t>(output));
      break;
    case kTfLiteInt16:
      reference_ops::Transpose(params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<int16_t>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<int16_t>(output));
      break;
    default:
      MicroPrintf(
          "Type %s is currently not supported by Transpose. "
          "Only float32, int8 and int16 are supported",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_TRANSPOSE() {
  return tflite::micro::RegisterOp(nullptr, TransposePrepare, TransposeEval);
}
}  // namespace tflite
