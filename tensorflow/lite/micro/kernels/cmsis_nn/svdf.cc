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

#include "tensorflow/lite/micro/kernels/svdf.h"

#include "Include/arm_nn_types.h"
#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/activation_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataSvdf));
}

TfLiteStatus EvalIntegerSVDF(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteEvalTensor* input_tensor,
                             const TfLiteEvalTensor* weights_feature_tensor,
                             const TfLiteEvalTensor* weights_time_tensor,
                             const TfLiteEvalTensor* bias_tensor,
                             const TfLiteSVDFParams* params,
                             TfLiteEvalTensor* activation_state_tensor,
                             TfLiteEvalTensor* output_tensor,
                             const OpDataSvdf& data) {
  cmsis_nn_dims input_dims;
  input_dims.n = input_tensor->dims->data[0];
  input_dims.h = input_tensor->dims->data[1];

  cmsis_nn_dims weights_feature_dims;
  weights_feature_dims.n = weights_feature_tensor->dims->data[0];
  weights_feature_dims.h = weights_feature_tensor->dims->data[1];

  cmsis_nn_dims weights_time_dims;
  weights_time_dims.n = weights_time_tensor->dims->data[0];
  weights_time_dims.h = weights_time_tensor->dims->data[1];

  cmsis_nn_dims bias_dims;
  bias_dims.n = bias_tensor->dims->data[0];

  cmsis_nn_dims state_dims;
  state_dims.n = bias_tensor->dims->data[0];
  state_dims.h = bias_tensor->dims->data[1];

  cmsis_nn_dims output_dims;
  output_dims.n = output_tensor->dims->data[0];
  output_dims.h = output_tensor->dims->data[1];

  cmsis_nn_svdf_params svdf_params;
  svdf_params.rank = params->rank;
  svdf_params.input_offset = data.input_zero_point;
  svdf_params.output_offset = data.output_zero_point;

  svdf_params.input_activation.min = INT16_MIN;
  svdf_params.input_activation.max = INT16_MAX;

  svdf_params.output_activation.min = INT8_MIN;
  svdf_params.output_activation.max = INT8_MAX;

  cmsis_nn_per_tensor_quant_params in_quant_params;
  in_quant_params.multiplier = data.effective_scale_1_a;
  in_quant_params.shift = data.effective_scale_1_b;

  cmsis_nn_per_tensor_quant_params out_quant_params;
  out_quant_params.multiplier = data.effective_scale_2_a;
  out_quant_params.shift = data.effective_scale_2_b;

  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(context->GetScratchBuffer != nullptr);

  cmsis_nn_context scratch_ctx;
  scratch_ctx.buf = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.scratch_tensor_index));

  cmsis_nn_context scratch_output_ctx;
  scratch_output_ctx.buf = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.scratch_output_tensor_index));

  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output_tensor);

  switch (weights_time_tensor->type) {
    case kTfLiteInt8: {
      arm_svdf_s8(
          &scratch_ctx, &scratch_output_ctx, &svdf_params, &in_quant_params,
          &out_quant_params, &input_dims,
          tflite::micro::GetTensorData<int8_t>(input_tensor), &state_dims,
          tflite::micro::GetTensorData<int8_t>(activation_state_tensor),
          &weights_feature_dims,
          tflite::micro::GetTensorData<int8_t>(weights_feature_tensor),
          &weights_time_dims,
          tflite::micro::GetTensorData<int8_t>(weights_time_tensor), &bias_dims,
          tflite::micro::GetTensorData<int32_t>(bias_tensor), &output_dims,
          output_data);
      return kTfLiteOk;
    }

    case kTfLiteInt16: {
      arm_svdf_state_s16_s8(
          &scratch_ctx, &scratch_output_ctx, &svdf_params, &in_quant_params,
          &out_quant_params, &input_dims,
          tflite::micro::GetTensorData<int8_t>(input_tensor), &state_dims,
          tflite::micro::GetTensorData<int16_t>(activation_state_tensor),
          &weights_feature_dims,
          tflite::micro::GetTensorData<int8_t>(weights_feature_tensor),
          &weights_time_dims,
          tflite::micro::GetTensorData<int16_t>(weights_time_tensor),
          &bias_dims, tflite::micro::GetTensorData<int32_t>(bias_tensor),
          &output_dims, output_data);
      return kTfLiteOk;
    }

    default:
      MicroPrintf("Could not find matching function for type %s.",
                  TfLiteTypeGetName(weights_time_tensor->type));
      return kTfLiteError;
  }
}

TfLiteStatus EvalSvdf(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataSvdf& data = *(static_cast<const OpDataSvdf*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsTimeTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 5)
          ? tflite::micro::GetEvalInput(context, node, kSvdfBiasTensor)
          : nullptr;
  TfLiteEvalTensor* activation_state = tflite::micro::GetMutableEvalInput(
      context, node, kSvdfInputActivationStateTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSvdfOutputTensor);

  switch (weights_time->type) {
    case kTfLiteFloat32: {
      EvalFloatSvdfReference(
          context, node, input, weights_feature, weights_time, bias, params,
          data.scratch_tensor_index, activation_state, output);
      return kTfLiteOk;
    }

    case kTfLiteInt8:
    case kTfLiteInt16: {
      return EvalIntegerSVDF(context, node, input, weights_feature,
                             weights_time, bias, params, activation_state,
                             output, data);
    }

    default:
      MicroPrintf("Type %s not currently supported.",
                  TfLiteTypeGetName(weights_feature->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalSvdfInt8(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataSvdf& data = *(static_cast<const OpDataSvdf*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsTimeTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 5)
          ? tflite::micro::GetEvalInput(context, node, kSvdfBiasTensor)
          : nullptr;
  TfLiteEvalTensor* activation_state = tflite::micro::GetMutableEvalInput(
      context, node, kSvdfInputActivationStateTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSvdfOutputTensor);

  TFLITE_DCHECK((weights_time->type == kTfLiteInt8) ||
                (weights_time->type == kTfLiteInt16));
  // Because of the TODO mentioned below, the int16 weight data type is not
  // split into a seperate registration.
  // TODO(#523): remove 16-bit code when no longer needed.
  return EvalIntegerSVDF(context, node, input, weights_feature, weights_time,
                         bias, params, activation_state, output, data);
}

}  // namespace

TfLiteRegistration Register_SVDF() {
  return tflite::micro::RegisterOp(Init, PrepareSvdf, EvalSvdf);
}

TfLiteRegistration Register_SVDF_INT8() {
  return tflite::micro::RegisterOp(Init, PrepareSvdf, EvalSvdfInt8);
}

}  // namespace tflite
