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
#include "tensorflow/lite/kernels/internal/reference/scatter_nd.h"

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

constexpr int kIndices = 0;
constexpr int kUpdates = 1;
constexpr int kShape = 2;
constexpr int kOutputTensor = 0;

template <typename IndicesT>
TfLiteStatus CheckShapes(TfLiteContext* context, const RuntimeShape& indices,
                         const RuntimeShape& updates,
                         const RuntimeShape& shape_shape,
                         const IndicesT* shape_data) {
  TF_LITE_ENSURE(context, (indices.DimensionsCount() >= 1) &&
                              (updates.DimensionsCount() >= 1) &&
                              (shape_shape.DimensionsCount() == 1));

  const int outer_dims = indices.DimensionsCount() - 1;
  for (int i = 0; i < outer_dims; ++i) {
    TF_LITE_ENSURE_EQ(context, indices.Dims(i), updates.Dims(i));
  }

  const int ix = indices.Dims(outer_dims);
  TF_LITE_ENSURE_EQ(context, updates.DimensionsCount() - outer_dims,
                    shape_shape.Dims(0) - ix);
  for (int i = 0; i + outer_dims < updates.DimensionsCount(); ++i) {
    TF_LITE_ENSURE_EQ(context, updates.Dims(i + outer_dims),
                      shape_data[ix + i]);
  }
  return kTfLiteOk;
}

TfLiteStatus ScatterNdPrepare(TfLiteContext* context, TfLiteNode* node) {

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* indices =
      micro_context->AllocateTempInputTensor(node, kIndices);
  TfLiteTensor* updates =
      micro_context->AllocateTempInputTensor(node, kUpdates);
  TfLiteTensor* shape =
      micro_context->AllocateTempInputTensor(node, kShape);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);

  switch (updates->type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt64:
    case kTfLiteInt32:
      break;
    default:
      MicroPrintf("Updates of type '%s' are not supported by scatter_nd.",
          TfLiteTypeGetName(updates->type));
      return kTfLiteError;
  }
  if (indices->type != shape->type) {
      MicroPrintf("Indices and shape must have the same type.");
    return kTfLiteError;
  }

  output->type = updates->type;


  TF_LITE_ENSURE_MSG(context, IsConstantTensor(shape),
                     "Non constant shape tensor not supported");

  switch (indices->type) {
    case kTfLiteInt32:
      // Omitting shape check seems to increase throughput even though this 
      // only takes place in Prepare
      TF_LITE_ENSURE_OK(
          context,
          CheckShapes<int32_t>(context, GetTensorShape(indices),
                                GetTensorShape(updates), GetTensorShape(shape),
                                GetTensorData<int32_t>(shape)));
      break;
    default:
      MicroPrintf("Indices of type '%s' are not supported by scatter_nd.",
          TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(indices);
  micro_context->DeallocateTempTfLiteTensor(updates);
  micro_context->DeallocateTempTfLiteTensor(shape);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;

}

template <typename IndicesT, typename UpdatesT>
TfLiteStatus ScatterNd(const TfLiteEvalTensor* indices, const TfLiteEvalTensor* updates,
                       TfLiteEvalTensor* output) {

  reference_ops::ScatterNd(tflite::micro::GetTensorShape(indices),
                           tflite::micro::GetTensorData<IndicesT>(indices),
                           tflite::micro::GetTensorShape(updates),
                           tflite::micro::GetTensorData<UpdatesT>(updates),
                           tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<UpdatesT>(output));
  return kTfLiteOk;
}

template <typename IndicesT>
TfLiteStatus EvalScatterNd(TfLiteContext* context, const TfLiteEvalTensor* indices,
                           const TfLiteEvalTensor* updates, TfLiteEvalTensor* output) {

  switch (updates->type) {
    case kTfLiteFloat32:
      return ScatterNd<IndicesT, float>(indices, updates, output);
    case kTfLiteInt8:
      return ScatterNd<IndicesT, int8_t>(indices, updates, output);
    default:
      MicroPrintf("Updates of type '%s' are not supported by scatter_nd.",
          TfLiteTypeGetName(updates->type));
      return kTfLiteError;
  }
}

TfLiteStatus ScatterNdEval(TfLiteContext* context, TfLiteNode* node) {

  const TfLiteEvalTensor* indices =
      tflite::micro::GetEvalInput(context, node, kIndices);
  const TfLiteEvalTensor* updates =
      tflite::micro::GetEvalInput(context, node, kUpdates);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (indices->type) {
    case kTfLiteInt32:
      return EvalScatterNd<int32_t>(context, indices, updates, output);
    default:
      MicroPrintf("Indices of type '%s' are not supported by scatter_nd.",
          TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }
}

} // namespace

TFLMRegistration Register_SCATTER_ND() {
  return tflite::micro::RegisterOp(nullptr, ScatterNdPrepare, ScatterNdEval);
}
}  // namespace tflite