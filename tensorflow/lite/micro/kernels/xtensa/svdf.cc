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

#include "tensorflow/lite/micro/kernels/svdf.h"

#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/activation_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_svdf.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

TfLiteStatus EvalIntegerSvdfHifi(TfLiteContext* context, TfLiteNode* node,
                                 const TfLiteEvalTensor* input_tensor,
                                 const TfLiteEvalTensor* weights_feature_tensor,
                                 const TfLiteEvalTensor* weights_time_tensor,
                                 const TfLiteEvalTensor* bias_tensor,
                                 const TfLiteSVDFParams* params,
                                 TfLiteEvalTensor* activation_state_tensor,
                                 TfLiteEvalTensor* output_tensor,
                                 const OpDataSvdf& data) {
  const int n_rank = params->rank;
  const int n_batch = input_tensor->dims->data[0];
  const int n_input = input_tensor->dims->data[1];
  const int n_filter = weights_feature_tensor->dims->data[0];
  const int n_unit = n_filter / n_rank;
  const int n_memory = weights_time_tensor->dims->data[1];

  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(context->GetScratchBuffer != nullptr);

  // Shift states.
  int16_t* const state_ptr =
      tflite::micro::GetTensorData<int16_t>(activation_state_tensor);

  // Left shift the activation_state.
  int num_bytes = sizeof(*state_ptr) * (n_batch * n_filter * n_memory - 1);
#if defined(HIFI5)
  memcpy(state_ptr, state_ptr + 1, num_bytes);
#else
  xa_nn_memmove_16(state_ptr, state_ptr + 1, (num_bytes >> 1));
#endif  // defined(HIFI5)

  // Note: no need to clear the latest activation, matmul is not accumulative.

  // Feature matmul.
  const int8_t* input = tflite::micro::GetTensorData<int8_t>(input_tensor);
  const int8_t* weight_feature =
      tflite::micro::GetTensorData<int8_t>(weights_feature_tensor);
  int16_t* result_in_batch = state_ptr + (n_memory - 1);

  for (int b = 0; b < n_batch; b++) {
    TF_LITE_ENSURE_EQ(context,
                      xa_nn_matXvec_out_stride_sym8sxasym8s_16(
                          &result_in_batch[b * n_filter * n_memory],
                          weight_feature, &input[b * n_input], NULL, n_filter,
                          n_input, n_input, n_memory, -data.input_zero_point,
                          (data.effective_scale_1_a), data.effective_scale_1_b),
                      0);
  }

  // Time weights dot product + activation
  for (int b = 0; b < n_batch; ++b) {
    const int16_t* vector1_ptr =
        tflite::micro::GetTensorData<int16_t>(weights_time_tensor);
    const int16_t* vector2_ptr =
        tflite::micro::GetTensorData<int16_t>(activation_state_tensor) +
        b * n_memory * n_filter;
    // TODO(#1751): account for optional bias tensor
    const int32_t* bias_ptr =
        tflite::micro::GetTensorData<int32_t>(bias_tensor);
    int8_t* output_ptr =
        tflite::micro::GetTensorData<int8_t>(output_tensor) + b * n_unit;

    // TODO(#1751): account for optional bias tensor
    TF_LITE_ENSURE_EQ(
        context,
        xa_nn_dot_prod_16x16_asym8s(
            output_ptr, vector1_ptr, vector2_ptr, bias_ptr, n_memory * n_rank,
            (data.effective_scale_2_a), data.effective_scale_2_b,
            data.output_zero_point, n_unit),
        0);
  }
  return kTfLiteOk;
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataSvdf));
}

TfLiteStatus PrepareInt8(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFIMINI) || defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params = static_cast<const TfLiteSVDFParams*>(node->builtin_data);

  // Validate Tensor Inputs (dtype depends on quantization):
  // [0] = Input, {2, batch_size, input_size}
  // [1] = Weights Feature, {2, num_filters, input_size}
  // [2] = Weights Time, {2, num_filters, memory_size}
  // [3] = Bias (optional), {1, num_units}
  // [4] = Activation State (variable),
  //         {2, batch_size, memory_size * num_filters}
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kSvdfInputTensor);
  TfLiteTensor* weights_feature =
      micro_context->AllocateTempInputTensor(node, kSvdfWeightsFeatureTensor);
  TfLiteTensor* weights_time =
      micro_context->AllocateTempInputTensor(node, kSvdfWeightsTimeTensor);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kSvdfBiasTensor);
  TfLiteTensor* activation_state = micro_context->AllocateTempInputTensor(
      node, kSvdfInputActivationStateTensor);

  // Define input constants based on input tensor definition above:
  const int rank = params->rank;
  const int input_size = input->dims->data[1];
  const int batch_size = input->dims->data[0];

#if defined(HIFIMINI)
  // Ensure the input size is a multiple of two.  This is necessary since
  // optimized kernels access the memory in chunks of two, and all accesses
  // must be aligned to 16 bits.
  // TODO(b/153202598): Remove when padding is allowed in TFLite tensors.
  TF_LITE_ENSURE_EQ(context, input_size % 2, 0);
#endif  // defined(HIFIMINI)

  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ENSURE_EQ(context, num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Validate Input Tensor:
  TF_LITE_ENSURE(context, input->type == kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 2);

  // Validate Tensor Output:
  // [0] = float/int8_t, {2, batch_size, num_units}
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kSvdfOutputTensor);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 2);
  TF_LITE_ENSURE_EQ(context, output->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, output->dims->data[1], num_units);

  // Validate Weights Feature Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights_feature), 2);
  TF_LITE_ENSURE_EQ(context, weights_feature->dims->data[1], input_size);

  // Validate Weights Time Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights_time), 2);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[0], num_filters);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[1], memory_size);

  // Validate Optional Bias Input Tensor:
  if (bias != nullptr) {
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], num_units);
    TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteInt32);
  }

  // Validate Activation State Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(activation_state), 2);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[1],
                    memory_size * num_filters);

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);
  TF_LITE_ENSURE_EQ(context, weights_feature->type, kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, weights_time->type, kTfLiteInt16);
  TF_LITE_ENSURE_EQ(context, activation_state->type, kTfLiteInt16);

  // Validate output tensor:
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);

  const double effective_scale_1 =
      static_cast<double>(input->params.scale * weights_feature->params.scale /
                          activation_state->params.scale);
  const double effective_scale_2 =
      static_cast<double>(activation_state->params.scale *
                          weights_time->params.scale / output->params.scale);

  // TODO(#1751): account for optional bias tensor
  TF_LITE_ENSURE_NEAR(context, static_cast<double>(bias->params.scale),
                      static_cast<double>(activation_state->params.scale *
                                          weights_time->params.scale),
                      1e-5);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataSvdf* data = static_cast<OpDataSvdf*>(node->user_data);

#if defined(HIFIMINI)
  QuantizeMultiplierForInt24(effective_scale_1, &data->effective_scale_1_a,
                             &data->effective_scale_1_b);
  QuantizeMultiplierForInt24(effective_scale_2, &data->effective_scale_2_a,
                             &data->effective_scale_2_b);
#else
  QuantizeMultiplier(effective_scale_1, &(data->effective_scale_1_a),
                     &(data->effective_scale_1_b));
  QuantizeMultiplier(effective_scale_2, &(data->effective_scale_2_a),
                     &(data->effective_scale_2_b));
#endif  // defined(HIFIMINI)

  data->input_zero_point = input->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
      context, batch_size * num_filters * sizeof(int32_t),
      &(data->scratch_tensor_index));
  TF_LITE_ENSURE_OK(context, scratch_status);
  const TfLiteStatus scratch_output_status =
      context->RequestScratchBufferInArena(
          context, batch_size * num_units * sizeof(int32_t),
          &(data->scratch_output_tensor_index));
  TF_LITE_ENSURE_OK(context, scratch_output_status);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(weights_time);
  micro_context->DeallocateTempTfLiteTensor(weights_feature);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(activation_state);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
#else
  return PrepareSvdf(context, node);
#endif  // defined(HIFIMINI) || defined(HIFI3) || defined(HIFI4) ||
        // defined(HIFI5)
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFIMINI) || defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kSvdfInputTensor);
  TfLiteTensor* weights_time =
      micro_context->AllocateTempInputTensor(node, kSvdfWeightsTimeTensor);

  TfLiteStatus status;
  if (input->type == kTfLiteInt8 && weights_time->type == kTfLiteInt16) {
    status = PrepareInt8(context, node);
  } else {
    status = PrepareSvdf(context, node);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(weights_time);

  return status;
#else
  return PrepareSvdf(context, node);
#endif  // defined(HIFIMINI) || defined(HIFI3) || defined(HIFI4) ||
        // defined(HIFI5)
}

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
  auto* params = static_cast<TfLiteSVDFParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsTimeTensor);
  // TODO(#1751): account for optional bias tensor
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 5)
          ? tflite::micro::GetEvalInput(context, node, kSvdfBiasTensor)
          : nullptr;
  TfLiteEvalTensor* activation_state = tflite::micro::GetMutableEvalInput(
      context, node, kSvdfInputActivationStateTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSvdfOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataSvdf& data = *(static_cast<const OpDataSvdf*>(node->user_data));

#if defined(HIFIMINI)
  return EvalIntegerSvdfHifimini(context, node, input, weights_feature,
                                 weights_time, bias, params, activation_state,
                                 output, data);
#elif defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  return EvalIntegerSvdfHifi(context, node, input, weights_feature,
                             weights_time, bias, params, activation_state,
                             output, data);
#else
  EvalInt16SvdfReference(context, node, input, weights_feature, weights_time,
                         bias, params, activation_state, output, data);
  return kTfLiteOk;
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = static_cast<TfLiteSVDFParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsTimeTensor);
  // TODO(#1751): account for optional bias tensor
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 5)
          ? tflite::micro::GetEvalInput(context, node, kSvdfBiasTensor)
          : nullptr;
  TfLiteEvalTensor* activation_state = tflite::micro::GetMutableEvalInput(
      context, node, kSvdfInputActivationStateTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSvdfOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataSvdf& data = *(static_cast<const OpDataSvdf*>(node->user_data));

  switch (weights_feature->type) {
    case kTfLiteFloat32: {
      EvalFloatSvdfReference(
          context, node, input, weights_feature, weights_time, bias, params,
          data.scratch_tensor_index, activation_state, output);
      break;
    }

    case kTfLiteInt8: {
      switch (weights_time->type) {
        case kTfLiteInt16: {
          return EvalInt8(context, node);
        }

        case kTfLiteInt8: {
          EvalInt8SvdfReference(context, node, input, weights_feature,
                                weights_time, bias, params, activation_state,
                                output, data);
          break;
        }

        default: {
          MicroPrintf("Type %s not currently supported.",
                      TfLiteTypeGetName(weights_time->type));
          return kTfLiteError;
        }
      }
      break;
    }

    default: {
      MicroPrintf("Type %s not currently supported.",
                  TfLiteTypeGetName(weights_feature->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_SVDF() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

TFLMRegistration Register_SVDF_INT8() {
  return tflite::micro::RegisterOp(Init, PrepareInt8, EvalInt8);
}

}  // namespace tflite
