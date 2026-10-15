/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/resize_nearest_neighbor.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {

namespace {

constexpr int kInputTensor = 0;
constexpr int kSizeTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* size =
      micro_context->AllocateTempInputTensor(node, kSizeTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);

  // Our current implementations rely on the input being 4D,
  // and the size being 1D tensor with exactly 2 elements.
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(size), 1);
  TF_LITE_ENSURE_EQ(context, size->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, size->dims->data[0], 2);

  output->type = input->type;

  if (!IsConstantTensor(size)) {
    MicroPrintf("Dynamic tensors are unsupported in tfmicro.");
    return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(size);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteResizeNearestNeighborParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* size =
      tflite::micro::GetEvalInput(context, node, kSizeTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  tflite::ResizeNearestNeighborParams op_params;
  op_params.align_corners = params->align_corners;
  op_params.half_pixel_centers = params->half_pixel_centers;

  if (output->type == kTfLiteFloat32) {
    reference_ops::ResizeNearestNeighbor(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int32_t>(input),
        tflite::micro::GetTensorShape(size),
        tflite::micro::GetTensorData<int32_t>(size),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int32_t>(output));
  } else if (output->type == kTfLiteInt8) {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))

  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input));   
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(size));      
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(output));

  int8_t *input_ptr = (int8_t *)(tflite::micro::GetTensorData<int8_t>(input));
  int8_t *output_ptr = (int8_t *)(tflite::micro::GetTensorData<int8_t>(output));
  int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32_t input_height = input_shape.Dims(1);
  int32_t input_width = input_shape.Dims(2);
  int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);
  int32_t output_height = tflite::micro::GetTensorData<int32_t>(size)[0];
  int32_t output_width = tflite::micro::GetTensorData<int32_t>(size)[1];

  const float height_scale =
      (op_params.align_corners && output_height > 1)
          ? (input_height - 1) / static_cast<float>(output_height - 1)
          : input_height / static_cast<float>(output_height);

  const float width_scale =
      (op_params.align_corners && output_width > 1)
          ? (input_width - 1) / static_cast<float>(output_width - 1)
          : input_width / static_cast<float>(output_width);
 
  const float offset = op_params.half_pixel_centers ? 0.5f : 0.0f;

  xa_nn_resize_nearest_neighbour_8_8(output_ptr, input_ptr, batches, input_height, input_width,
                    depth, batches, output_height, output_width, depth, height_scale, width_scale,
                    offset, offset, op_params.align_corners);

#else    
    reference_ops::ResizeNearestNeighbor(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(size),
        tflite::micro::GetTensorData<int32_t>(size),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
#endif        
  } else if (output->type == kTfLiteInt16) {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))

  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(input));   
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(size));      
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, tflite::micro::GetTensorShape(output));

  int16_t *input_ptr = (int16_t *)(tflite::micro::GetTensorData<int16_t>(input));
  int16_t *output_ptr = (int16_t *)(tflite::micro::GetTensorData<int16_t>(output));
  int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32_t input_height = input_shape.Dims(1);
  int32_t input_width = input_shape.Dims(2);
  int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);
  int32_t output_height = tflite::micro::GetTensorData<int32_t>(size)[0];
  int32_t output_width = tflite::micro::GetTensorData<int32_t>(size)[1];

  const float height_scale =
      (op_params.align_corners && output_height > 1)
          ? (input_height - 1) / static_cast<float>(output_height - 1)
          : input_height / static_cast<float>(output_height);

  const float width_scale =
      (op_params.align_corners && output_width > 1)
          ? (input_width - 1) / static_cast<float>(output_width - 1)
          : input_width / static_cast<float>(output_width);
 
  const float offset = op_params.half_pixel_centers ? 0.5f : 0.0f;

  xa_nn_resize_nearest_neighbour_16_16(output_ptr, input_ptr, batches, input_height, input_width,
                    depth, batches, output_height, output_width, depth, height_scale, width_scale,
                    offset, offset, op_params.align_corners);

#else    
    reference_ops::ResizeNearestNeighbor(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorShape(size),
        tflite::micro::GetTensorData<int32_t>(size),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
#endif        
  } else {
    MicroPrintf("Output tensor type %s (%d) not supported.",
                TfLiteTypeGetName(output->type), output->type);

    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_RESIZE_NEAREST_NEIGHBOR() {
  return tflite::micro::RegisterOp(nullptr, Prepare, Eval);
}

}  // namespace tflite
