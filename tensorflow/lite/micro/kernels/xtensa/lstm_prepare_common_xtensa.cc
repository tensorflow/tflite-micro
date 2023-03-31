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

#if defined(HIFI4) || defined(HIFI5)

#include <cmath>
#include <cstddef>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/micro_tensor_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_lstm.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

TfLiteStatus PrecomputeZeroPointTimesWeightWithBias(
    TfLiteContext* context, int32_t zero_point,
    const TfLiteTensor* weight_tensor, const TfLiteTensor* bias_tensor,
    int32_t** output) {
  if (weight_tensor == nullptr) {
    return kTfLiteOk;
  }

  const RuntimeShape& weight_shape = GetTensorShape(weight_tensor);
  TF_LITE_ENSURE_EQ(context, weight_shape.DimensionsCount(), 2);
  const int row = weight_shape.Dims(0);
  const int col = weight_shape.Dims(1);
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  *output = static_cast<int32_t*>(
      context->AllocatePersistentBuffer(context, row * sizeof(int32_t)));

  if (bias_tensor == nullptr) {
    std::memset(*output, 0, row * sizeof(int32_t));
  } else {
    const int32_t* bias = GetTensorData<int32_t>(bias_tensor);
    std::memcpy(*output, bias, row * sizeof(int32_t));
  }
  if (zero_point != 0) {
    const int8_t* weight = GetTensorData<int8_t>(weight_tensor);
    tflite::tensor_utils::MatrixScalarMultiplyAccumulate(weight, zero_point,
                                                         row, col, *output);
  }
  return kTfLiteOk;
}

TfLiteStatus PopulatePrecomputedZPTimesWeightsWithBias(
    TfLiteContext* context, lstm_xtensa::XtensaOpDataLstm* op_data,
    TfLiteNode* node, const LstmTensors& lstm_tensors) {
  const TfLiteTensor* input = lstm_tensors.GetInternalTensor(kLstmInputTensor);
  const TfLiteTensor* output_state =
      lstm_tensors.GetInternalTensor(kLstmOutputStateTensor);

  const int32_t input_zero_point = -input->params.zero_point;
  const int32_t output_state_zero_point = -output_state->params.zero_point;

  const TfLiteTensor* input_to_input_weights =
      lstm_tensors.GetInternalTensor(kLstmInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights =
      lstm_tensors.GetInternalTensor(kLstmInputToForgetWeightsTensor);
  const TfLiteTensor* input_to_cell_weights =
      lstm_tensors.GetInternalTensor(kLstmInputToCellWeightsTensor);
  const TfLiteTensor* input_to_output_weights =
      lstm_tensors.GetInternalTensor(kLstmInputToOutputWeightsTensor);

  const TfLiteTensor* recurrent_to_input_weights =
      lstm_tensors.GetInternalTensor(kLstmRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights =
      lstm_tensors.GetInternalTensor(kLstmRecurrentToForgetWeightsTensor);
  const TfLiteTensor* recurrent_to_cell_weights =
      lstm_tensors.GetInternalTensor(kLstmRecurrentToCellWeightsTensor);
  const TfLiteTensor* recurrent_to_output_weights =
      lstm_tensors.GetInternalTensor(kLstmRecurrentToOutputWeightsTensor);

  lstm_xtensa::IntegerLstmParameter* integer_lstm_params =
      &op_data->integer_lstm_param;

  // Get bias and perform zero point calculation.
  // When there is layer normalization, the gate bias does not apply to matmul
  // directly:
  //      y = ln(w * x + w * r + w * c) + b.

  // Forget gate.
  const TfLiteTensor* forget_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmForgetGateBiasTensor);
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, input_zero_point, input_to_forget_weights, forget_gate_bias,
          &(integer_lstm_params->input_to_forget_effective_bias)));

  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, output_state_zero_point, recurrent_to_forget_weights,
          nullptr, &(integer_lstm_params->recurrent_to_forget_effective_bias)));

  // Modulation gate.
  const TfLiteTensor* cell_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmCellGateBiasTensor);
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, input_zero_point, input_to_cell_weights, cell_gate_bias,
          &(integer_lstm_params->input_to_cell_effective_bias)));
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, output_state_zero_point, recurrent_to_cell_weights, nullptr,
          &(integer_lstm_params->recurrent_to_cell_effective_bias)));

  // Output gate.
  const TfLiteTensor* output_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmOutputGateBiasTensor);
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, input_zero_point, input_to_output_weights, output_gate_bias,
          &(integer_lstm_params->input_to_output_effective_bias)));

  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, output_state_zero_point, recurrent_to_output_weights,
          nullptr, &(integer_lstm_params->recurrent_to_output_effective_bias)));

  // Input gate. The calculation is only meaningful for non-cifg case.
  const TfLiteTensor* input_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmInputGateBiasTensor);
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, input_zero_point, input_to_input_weights, input_gate_bias,
          &(integer_lstm_params->input_to_input_effective_bias)));
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, output_state_zero_point, recurrent_to_input_weights, nullptr,
          &(integer_lstm_params->recurrent_to_input_effective_bias)));

  return kTfLiteOk;
}

}  // namespace

// Resize the output and  state tensors based on the sizes of the input tensors.
// Allocate a temporary scratch tensor. Also check that the sizes of the input
// tensors match each other.
TfLiteStatus XtensaUnidirectionalSequenceLstmPrepareInt8x8_16(
    TfLiteContext* context, TfLiteNode* node, const LstmTensors& lstm_tensors) {
  lstm_xtensa::XtensaOpDataLstm* op_data =
      static_cast<lstm_xtensa::XtensaOpDataLstm*>(node->user_data);

  const int n_batch = op_data->size_info.batch_size;
  const int n_cell = op_data->size_info.state_dimension;

  const TfLiteTensor* input_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmInputGateBiasTensor);
  TF_LITE_ENSURE_TYPES_EQ(context, input_gate_bias->type, kTfLiteInt32);

  const TfLiteTensor* forget_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmForgetGateBiasTensor);
  TF_LITE_ENSURE_TYPES_EQ(context, forget_gate_bias->type, kTfLiteInt32);

  const TfLiteTensor* cell_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmCellGateBiasTensor);
  TF_LITE_ENSURE_TYPES_EQ(context, cell_gate_bias->type, kTfLiteInt32);

  const TfLiteTensor* output_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmOutputGateBiasTensor);
  TF_LITE_ENSURE_TYPES_EQ(context, output_gate_bias->type, kTfLiteInt32);

  // Allocate 5th scratch buffer. Need 4 16-bit buffer with size n_batch *
  // n_cell and 1 8-bit buffer with size n_batch * n_cell. For integer
  // UnidirectionalSequenceLSTM, we do not need the extra 32-bit buffer.
  TF_LITE_ENSURE_OK(
      context, context->RequestScratchBufferInArena(
                   context, n_batch * n_cell * TfLiteTypeGetSize(kTfLiteInt8),
                   &(op_data->scratch_index_4)));

  // Populate precomputed zp * weight.
  TF_LITE_ENSURE_OK(context, PopulatePrecomputedZPTimesWeightsWithBias(
                                 context, op_data, node, lstm_tensors));

  return kTfLiteOk;
}

}  // namespace tflite

#endif  // defined(HIFI4) || defined(HIFI5)
