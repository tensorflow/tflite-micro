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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_fully_connected.h"

namespace tflite {

TfLiteStatus XtensaEvalFullyConnectedF32(
    TfLiteContext* context, TfLiteNode* node, const OpDataFullyConnected& data,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
#if !defined(VISION_P6)

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node,
                                              kFullyConnectedWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kFullyConnectedBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  const float* bias_data =
#ifdef USE_TFLM_COMPRESSION
      tflite::micro::GetOptionalTensorData<float>(
          micro_context, bias, bias_comp_td, data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
      tflite::micro::GetOptionalTensorData<float>(bias);
#endif  // USE_TFLM_COMPRESSION

  const float* filter_data =
#ifdef USE_TFLM_COMPRESSION
      tflite::micro::GetTensorData<float>(
          micro_context, filter, weights_comp_td, data.weights_scratch_index);
#else   // USE_TFLM_COMPRESSION
      tflite::micro::GetTensorData<float>(filter);
#endif  // USE_TFLM_COMPRESSION

#endif  // !defined(VISION_P6)

  const auto* node_data =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
  const FullyConnectedParams params =
      FullyConnectedParamsFloat(node_data->activation);
  const float* input_data = tflite::micro::GetTensorData<float>(input);
  float* output_data = tflite::micro::GetTensorData<float>(output);

#if HIFI_VFPU && (defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int num_batches =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  const int output_depth =
      output_shape.Dims(output_shape.DimensionsCount() - 1);

  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  if (num_batches == 1) {
    TF_LITE_ENSURE_EQ(
        context,
        xa_nn_fully_connected_f32(output_data, filter_data, input_data,
                                  bias_data, accum_depth, output_depth),
        0);
  } else {
    TF_LITE_ENSURE_EQ(context,
                      xa_nn_matmul_f32xf32_f32(
                          output_data, filter_data, input_data, bias_data,
                          output_depth, accum_depth, accum_depth, num_batches,
                          accum_depth, output_depth, 1),
                      0);
  }
  TF_LITE_ENSURE_EQ(
      context,
      xa_nn_vec_activation_min_max_f32_f32(
          output_data, output_data, params.float_activation_min,
          params.float_activation_max, num_batches * output_depth),
      0);
#else
  tflite::reference_ops::FullyConnected(
      params, tflite::micro::GetTensorShape(input), input_data,
      tflite::micro::GetTensorShape(filter), filter_data,
      tflite::micro::GetTensorShape(bias), bias_data,
      tflite::micro::GetTensorShape(output), output_data);
#endif  // HIFI_VFPU && (defined(HIFI3) || defined(HIFI4) || defined(HIFI5))

  return kTfLiteOk;
}

namespace {

TfLiteStatus EvalFloat32(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data =
      *(static_cast<const OpDataFullyConnected*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  return XtensaEvalFullyConnectedF32(context, node, data, input, filter, bias,
                                     output);
}

}  // namespace

TFLMRegistration Register_FULLY_CONNECTED_FLOAT32() {
  return tflite::micro::RegisterOp(XtensaInitFullyConnected,
                                   XtensaPrepareFullyConnected, EvalFloat32);
}

}  // namespace tflite
