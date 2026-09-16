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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_fully_connected.h"

namespace tflite {

TfLiteStatus XtensaEvalFullyConnectedQuantizedFloat32(
    TfLiteContext* context, TfLiteNode* node, const OpDataFullyConnected& data,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
  const float32_t* bias_data =
      nullptr != bias ? tflite::micro::GetTensorData<float32_t>(bias) : nullptr;
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int num_batches =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  const int output_depth =
      output_shape.Dims(output_shape.DimensionsCount() - 1);

  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  FullyConnectedParams op_params = FullyConnectedParamsFloat(params->activation);
#ifndef HIFI_IQ
  if(num_batches == 1) {
      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_fully_connected_f32(
              tflite::micro::GetTensorData<float32_t>(output),
              tflite::micro::GetTensorData<float32_t>(filter),
              tflite::micro::GetTensorData<float32_t>(input),
              bias_data, accum_depth, output_depth),
          0);
  }
  else{
      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_matmul_f32xf32_f32(
              tflite::micro::GetTensorData<float32_t>(output),
              tflite::micro::GetTensorData<float32_t>(filter),
              tflite::micro::GetTensorData<float32_t>(input),
              bias_data, output_depth, accum_depth, accum_depth,
              num_batches, accum_depth, output_depth, 1),
          0);    
  }
  float32_t* output_arr = tflite::micro::GetTensorData<float32_t>(output);
  TF_LITE_ENSURE_EQ(
      context,
      xa_nn_vec_activation_min_max_f32_f32(
          output_arr, output_arr, op_params.float_activation_min,
          op_params.float_activation_max, num_batches * output_depth),
      0);
#else
  if(num_batches == 1) {
      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_fully_connected_v2_f32(
              tflite::micro::GetTensorData<float32_t>(output),
              tflite::micro::GetTensorData<float32_t>(filter),
              tflite::micro::GetTensorData<float32_t>(input),
              bias_data, accum_depth, output_depth,
              op_params.float_activation_min, op_params.float_activation_max, NULL),
          0);
  }
  else{
      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_matmul_v2_f32xf32_f32(
              tflite::micro::GetTensorData<float32_t>(output),
              tflite::micro::GetTensorData<float32_t>(filter),
              tflite::micro::GetTensorData<float32_t>(input),
              bias_data, output_depth, accum_depth, accum_depth,
              num_batches, accum_depth, output_depth, 1,
              op_params.float_activation_min, op_params.float_activation_max, NULL),
          0);    
  }
#endif // HIFI_IQ

#else

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node,
                                              kFullyConnectedWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kFullyConnectedBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  tflite::reference_ops::FullyConnected(
      FullyConnectedParamsFloat(params->activation),
      tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<float>(input),
      tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
      tflite::micro::GetTensorData<float>(micro_context, filter,
                                          weights_comp_td,
                                          data.weights_scratch_index),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetOptionalTensorData<float>(
          micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
      tflite::micro::GetTensorData<float>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetOptionalTensorData<float>(bias),
#endif  // USE_TFLM_COMPRESSION
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output));
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
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

  return XtensaEvalFullyConnectedQuantizedFloat32(context, node, data, input,
                                               filter, bias, output);
}

}  // namespace

TFLMRegistration Register_FULLY_CONNECTED_FLOAT32() {
  return tflite::micro::RegisterOp(XtensaInitFullyConnected,
                                   XtensaPrepareFullyConnected, EvalFloat32);
}

}  // namespace tflite
