/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/broadcast_to.h"

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace tflite {

namespace {
constexpr int kInputTensor = 0;
constexpr int kShapeTensor = 1;
constexpr int kOutputTensor = 0;
// A maximum of 5 dimensions.
constexpr int kMaxDims = 5;

struct BroadcastToContext {
  BroadcastToContext(TfLiteContext* context, TfLiteNode* node) {
    micro_context = GetMicroContext(context);
    input = micro_context->AllocateTempInputTensor(node, kInputTensor);
    shape = micro_context->AllocateTempInputTensor(node, kShapeTensor);
    output = micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  }
  ~BroadcastToContext() {
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(shape);
    micro_context->DeallocateTempTfLiteTensor(output);
  }
  MicroContext* micro_context;
  TfLiteTensor* input;
  TfLiteTensor* shape;
  TfLiteTensor* output;
};

TfLiteStatus ValidateOutputTensor(TfLiteContext* context,
                                  BroadcastToContext* op_context) {
  // Ensures the shape is 1D tensor.
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->shape), 1);

  // Ensure output dims is not less than input dims.
  int input_num_dims = NumDimensions(op_context->input);
  int output_num_dims = NumDimensions(op_context->output);
  int shape_num_dims = SizeOfDimension(op_context->shape, 0);
  TF_LITE_ENSURE_MSG(context, output_num_dims == shape_num_dims,
                     "Output must match with the expected shape dimension.");
  TF_LITE_ENSURE_MSG(context, input_num_dims <= output_num_dims,
                     "Output shape must be broadcastable from input shape.");
  TF_LITE_ENSURE_MSG(context, output_num_dims <= kMaxDims,
                     "BroadcastTo only supports 1-5D tensor.");

  // Check if output shape is broadcastable from input shape.
  auto get_shape_data = [op_context](int i) -> int32_t {
    if (op_context->shape->type == kTfLiteInt32) {
      return GetTensorData<int32_t>(op_context->shape)[i];
    } else {
      return GetTensorData<int64_t>(op_context->shape)[i];
    }
  };

  int extending_dims = output_num_dims - input_num_dims;
  for (int idx = 0; idx < input_num_dims; ++idx) {
    TF_LITE_ENSURE_MSG(context,
                       (SizeOfDimension(op_context->input, idx) == 1 ||
                        SizeOfDimension(op_context->input, idx) ==
                            get_shape_data(extending_dims + idx)),
                       "Output shape must be broadcastable from input shape.");
  }

  // Validating the shape of the output tensor.
  tflite::RuntimeShape output_shape =
      tflite::GetTensorShape(op_context->output);
  for (int idx = 0; idx < output_num_dims; ++idx) {
    TF_LITE_ENSURE(context, output_shape.Dims(idx) == get_shape_data(idx));
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  BroadcastToContext op_context(context, node);

  TF_LITE_ENSURE_MSG(context, (NumDimensions(op_context.input) <= kMaxDims),
                     "BroadcastTo only supports 1-5D tensor.");

  TF_LITE_ENSURE(context, op_context.shape->type == kTfLiteInt32 ||
                              op_context.shape->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  // Not yet support String type due to the use of memcopy with fixed size.
  TF_LITE_ENSURE(context, op_context.input->type != kTfLiteString);

  return ValidateOutputTensor(context, &op_context);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output = micro::GetEvalOutput(context, node, kOutputTensor);

  // BroadcastTo op support upto 8 dims, matching the support of Tensorflow.
  reference_ops::BroadcastTo<kMaxDims>(
      micro::GetTensorShape(input), input->data.raw,
      micro::GetTensorShape(output), output->data.raw, input->type);
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration Register_BROADCAST_TO() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite