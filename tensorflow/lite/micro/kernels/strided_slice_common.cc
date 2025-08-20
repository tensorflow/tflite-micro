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
#include <cmath>
#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/strided_slice.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

struct StridedSliceContext {
  StridedSliceContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteStridedSliceParams*>(node->builtin_data);
    micro_context = GetMicroContext(context);
    input =
        micro_context->AllocateTempInputTensor(node, kStridedSliceInputTensor);
    begin =
        micro_context->AllocateTempInputTensor(node, kStridedSliceBeginTensor);
    end = micro_context->AllocateTempInputTensor(node, kStridedSliceEndTensor);
    strides = micro_context->AllocateTempInputTensor(
        node, kStridedSliceStridesTensor);
    output = micro_context->AllocateTempOutputTensor(node,
                                                     kStridedSliceOutputTensor);
    dims = NumDimensions(input);
  }
  ~StridedSliceContext() {
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(begin);
    micro_context->DeallocateTempTfLiteTensor(end);
    micro_context->DeallocateTempTfLiteTensor(strides);
    micro_context->DeallocateTempTfLiteTensor(output);
  }
  const TfLiteStridedSliceParams* params;
  MicroContext* micro_context;
  TfLiteTensor* input;
  TfLiteTensor* begin;
  TfLiteTensor* end;
  TfLiteTensor* strides;
  TfLiteTensor* output;
  int dims;
};

// This Op only supports 1-4D cases and since we use the reference 4D
// implementation, the 1-3D tensors are mapped to 4D.
const int kMaxDim = 4;

tflite::StridedSliceParams BuildStridedSliceParams(
    StridedSliceContext* op_context) {
  tflite::StridedSliceParams op_params{};
  op_params.start_indices_count = op_context->dims;
  op_params.stop_indices_count = op_context->dims;
  op_params.strides_count = op_context->dims;

  for (int i = 0; i < op_context->dims; ++i) {
    op_params.start_indices[i] = GetTensorData<int32_t>(op_context->begin)[i];
    op_params.stop_indices[i] = GetTensorData<int32_t>(op_context->end)[i];
    op_params.strides[i] = GetTensorData<int32_t>(op_context->strides)[i];
  }

  op_params.begin_mask = op_context->params->begin_mask;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = op_context->params->end_mask;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = op_context->params->shrink_axis_mask;
  return op_params;
}

// Processes the indexing tensors (begin, end and strides) to resize the
// output tensor. This function is callable from both Prepare() and Eval() as
// long as the caller ensures the indexing tensors are present.
TfLiteStatus CheckOutputSize(TfLiteContext* context,
                             StridedSliceContext* op_context) {
  using ::tflite::strided_slice::StartForAxis;
  using ::tflite::strided_slice::StopForAxis;
  TfLiteIntArray* output_shape = op_context->output->dims;
  int shape_size = 0;
  auto op_params = BuildStridedSliceParams(op_context);
  auto input_shape = GetTensorShape(op_context->input);
  for (int idx = 0; idx < op_context->dims; ++idx) {
    int32_t stride = GetTensorData<int32_t>(op_context->strides)[idx];
    TF_LITE_ENSURE_MSG(context, stride != 0, "stride value has to be non-zero");
    int32_t begin = StartForAxis(op_params, input_shape, idx);
    int32_t end = StopForAxis(op_params, input_shape, idx, begin);

    // When shrinking an axis, the end position does not matter (and can be
    // incorrect when negative indexing is used, see Issue #19260). Always use
    // begin + 1 to generate a length 1 slice, since begin has
    // already been adjusted for negative indices by StartForAxis.
    const bool shrink_axis = op_context->params->shrink_axis_mask & (1 << idx);
    if (shrink_axis) {
      end = begin + 1;
    }

    // This is valid for both positive and negative strides
    int32_t dim_shape = std::ceil((end - begin) / static_cast<float>(stride));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    if (!shrink_axis) {
      TF_LITE_ENSURE_EQ(context, output_shape->data[shape_size], dim_shape);
      shape_size++;
    }
  }
  TF_LITE_ENSURE_EQ(context, output_shape->size, shape_size);
  return kTfLiteOk;
}

}  // namespace

void* StridedSliceInit(TfLiteContext* context, const char* buffer,
                       size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(StridedSliceParams));
}

TfLiteStatus StridedSlicePrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  StridedSliceParams* op_params =
      static_cast<StridedSliceParams*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  StridedSliceContext op_context(context, node);
  TF_LITE_ENSURE_MSG(context, op_context.dims <= kMaxDim,
                     "input dim should not exceed 4");
  TF_LITE_ENSURE_MSG(context, IsConstantTensor(op_context.begin),
                     "Non-constant >begin< tensor is not supported");
  TF_LITE_ENSURE_MSG(context, IsConstantTensor(op_context.end),
                     "Non-constant >end< tensor is not supported");
  TF_LITE_ENSURE_MSG(context, IsConstantTensor(op_context.strides),
                     "Non-constant >strides< tensor is not supported");
  auto params = BuildStridedSliceParams(&op_context);
  memcpy(op_params, &params, sizeof(StridedSliceParams));
  return CheckOutputSize(context, &op_context);
}

}  // namespace tflite
