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

// Integer version of unidirectional sequence LSTM. Only the standard LSTM
// (defined in the keras LSTM layer, e.g., no peephole etc.) is supported here.
// Currently used by the 8 bits activation case only, except for fallbacks.

#include <algorithm>
#include <limits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/micro_tensor_utils.h"

namespace tflite {

namespace {

struct OpData {
  OpDataLSTM params_ref;
  cmsis_nn_lstm_params params_cmsis_nn;
};

/*Helper Functions*/
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
    memset(*output, 0, row * sizeof(int32_t));
  } else {
    const int32_t* bias = GetTensorData<int32_t>(bias_tensor);
    memcpy(*output, bias, row * sizeof(int32_t));
  }

  if (zero_point != 0) {
    const int8_t* weight = GetTensorData<int8_t>(weight_tensor);
    tflite::tensor_utils::MatrixScalarMultiplyAccumulate(weight, zero_point,
                                                         row, col, *output);
  }
  return kTfLiteOk;
}

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             const LstmTensors& lstm_tensors, OpData* op_data) {
  const TfLiteTensor* input = lstm_tensors.GetInternalTensor(kLstmInputTensor);
  const TfLiteTensor* output_state =
      lstm_tensors.GetInternalTensor(tflite::kLstmOutputStateTensor);

  TF_LITE_ENSURE(context, input->type == kTfLiteInt8);

  op_data->params_cmsis_nn.output_state_offset =
      output_state->params.zero_point;

  const TfLiteTensor* input_to_forget_weights =
      lstm_tensors.GetInternalTensor(kLstmInputToForgetWeightsTensor);
  const TfLiteTensor* input_to_input_weights =
      lstm_tensors.GetInternalTensor(kLstmInputToInputWeightsTensor);
  const TfLiteTensor* input_to_output_weights =
      lstm_tensors.GetInternalTensor(kLstmInputToOutputWeightsTensor);
  const TfLiteTensor* input_to_cell_weights =
      lstm_tensors.GetInternalTensor(kLstmInputToCellWeightsTensor);
  const TfLiteTensor* forget_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmForgetGateBiasTensor);
  const TfLiteTensor* cell_state =
      lstm_tensors.GetInternalTensor(kLstmCellStateTensor);

  const TfLiteTensor* cell_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmCellGateBiasTensor);
  const TfLiteTensor* output_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmOutputGateBiasTensor);
  const TfLiteTensor* input_gate_bias =
      lstm_tensors.GetInternalTensor(kLstmInputGateBiasTensor);
  const TfLiteTensor* recurrent_to_forget_weights =
      lstm_tensors.GetInternalTensor(kLstmRecurrentToForgetWeightsTensor);
  const TfLiteTensor* recurrent_to_cell_weights =
      lstm_tensors.GetInternalTensor(kLstmRecurrentToCellWeightsTensor);
  const TfLiteTensor* recurrent_to_output_weights =
      lstm_tensors.GetInternalTensor(kLstmRecurrentToOutputWeightsTensor);
  const TfLiteTensor* recurrent_to_input_weights =
      lstm_tensors.GetInternalTensor(kLstmRecurrentToInputWeightsTensor);
  const TfLiteTensor* cell_to_output_weights =
      lstm_tensors.GetInternalTensor(kLstmCellToOutputWeightsTensor);
  const TfLiteTensor* forget_layer_norm_coefficients =
      lstm_tensors.GetInternalTensor(kLstmForgetLayerNormCoefficientsTensor);
  const TfLiteTensor* projection_weights =
      lstm_tensors.GetInternalTensor(kLstmProjectionWeightsTensor);

  const bool use_layer_norm = (forget_layer_norm_coefficients != nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);
  const bool use_projection = (projection_weights != nullptr);
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool lstm_unsupported_config =
      use_layer_norm || use_peephole || use_projection || use_cifg;
  TFLITE_DCHECK(!lstm_unsupported_config);

  // Pre-calculate bias + zero_point * weight.
  int32_t* input_to_forget_effective_bias = nullptr;
  int32_t* recurrent_to_forget_effective_bias = nullptr;
  int32_t* input_to_cell_effective_bias = nullptr;
  int32_t* recurrent_to_cell_effective_bias = nullptr;
  int32_t* input_to_output_effective_bias = nullptr;
  int32_t* recurrent_to_output_effective_bias = nullptr;
  int32_t* input_to_input_effective_bias = nullptr;
  int32_t* recurrent_to_input_effective_bias = nullptr;

  const int32_t output_state_zero_point =
      -op_data->params_cmsis_nn.output_state_offset;
  const int32_t input_zero_point = -input->params.zero_point;

  TF_LITE_ENSURE_OK(context,
                    PrecomputeZeroPointTimesWeightWithBias(
                        context, input_zero_point, input_to_forget_weights,
                        forget_gate_bias, &input_to_forget_effective_bias));

  TF_LITE_ENSURE_OK(context, PrecomputeZeroPointTimesWeightWithBias(
                                 context, output_state_zero_point,
                                 recurrent_to_forget_weights, nullptr,
                                 &recurrent_to_forget_effective_bias));

  // Modulation gate.
  TF_LITE_ENSURE_OK(context,
                    PrecomputeZeroPointTimesWeightWithBias(
                        context, input_zero_point, input_to_cell_weights,
                        cell_gate_bias, &input_to_cell_effective_bias));
  TF_LITE_ENSURE_OK(
      context, PrecomputeZeroPointTimesWeightWithBias(
                   context, output_state_zero_point, recurrent_to_cell_weights,
                   nullptr, &recurrent_to_cell_effective_bias));

  // Output gate.
  TF_LITE_ENSURE_OK(context,
                    PrecomputeZeroPointTimesWeightWithBias(
                        context, input_zero_point, input_to_output_weights,
                        output_gate_bias, &input_to_output_effective_bias));

  TF_LITE_ENSURE_OK(context, PrecomputeZeroPointTimesWeightWithBias(
                                 context, output_state_zero_point,
                                 recurrent_to_output_weights, nullptr,
                                 &recurrent_to_output_effective_bias));

  // Input gate. The calculation is only meaningful for non-cifg case.
  TF_LITE_ENSURE_OK(context,
                    PrecomputeZeroPointTimesWeightWithBias(
                        context, input_zero_point, input_to_input_weights,
                        input_gate_bias, &input_to_input_effective_bias));
  TF_LITE_ENSURE_OK(
      context, PrecomputeZeroPointTimesWeightWithBias(
                   context, output_state_zero_point, recurrent_to_input_weights,
                   nullptr, &recurrent_to_input_effective_bias));

  op_data->params_cmsis_nn.i2f_effective_bias = input_to_forget_effective_bias;
  op_data->params_cmsis_nn.r2f_effective_bias =
      recurrent_to_forget_effective_bias;
  op_data->params_cmsis_nn.i2c_effective_bias = input_to_cell_effective_bias;
  op_data->params_cmsis_nn.r2c_effective_bias =
      recurrent_to_cell_effective_bias;
  op_data->params_cmsis_nn.i2o_effective_bias = input_to_output_effective_bias;
  op_data->params_cmsis_nn.r2o_effective_bias =
      recurrent_to_output_effective_bias;
  op_data->params_cmsis_nn.i2i_effective_bias = input_to_input_effective_bias;
  op_data->params_cmsis_nn.r2i_effective_bias =
      recurrent_to_input_effective_bias;

  // Get intermediate scales and zero points.
  float intermediate_scale[5];
  int32_t intermediate_zp[5];
  for (int i = 0; i < 4; ++i) {
    // Q3.12 for activation functions.
    intermediate_scale[i] = std::pow(2.0f, -12.0f);
    intermediate_zp[i] = 0;
  }

  MicroContext* micro_context = GetMicroContext(context);
  // In the absence of projection, hidden becomes otuput and this intermediate
  // is ignored.
  TfLiteTensor* hidden = micro_context->AllocateTempIntermediateTensor(node, 4);
  TF_LITE_ENSURE(context, hidden->quantization.type != kTfLiteNoQuantization);
  auto* hidden_params =
      static_cast<TfLiteAffineQuantization*>(hidden->quantization.params);
  intermediate_scale[4] = hidden_params->scale->data[0];
  intermediate_zp[4] = hidden_params->zero_point->data[0];
  if (hidden != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(hidden);
  }

  // Scales.
  const float default_scale = 1.0;
  float input_scale = default_scale;
  float input_to_input_weight_scale = default_scale;
  float recurrent_to_input_weight_scale = default_scale;
  float input_to_forget_weight_scale = default_scale;
  float recurrent_to_forget_weight_scale = default_scale;
  float input_to_cell_weight_scale = default_scale;
  float recurrent_to_cell_weight_scale = default_scale;
  float input_to_output_weight_scale = default_scale;
  float recurrent_to_output_weight_scale = default_scale;
  float output_state_scale = default_scale;
  int cell_scale = 1;

  // Effective scales.
  float effective_input_to_input_scale = default_scale;
  float effective_recurrent_to_input_scale = default_scale;
  float effective_cell_to_input_scale = default_scale;
  float effective_input_to_forget_scale = default_scale;
  float effective_recurrent_to_forget_scale = default_scale;
  float effective_cell_to_forget_scale = default_scale;
  float effective_input_to_cell_scale = default_scale;
  float effective_recurrent_to_cell_scale = default_scale;
  float effective_input_to_output_scale = default_scale;
  float effective_recurrent_to_output_scale = default_scale;
  float effective_cell_to_output_scale = default_scale;
  float effective_hidden_scale = default_scale;

  // Populate scales.
  input_to_input_weight_scale = input_to_input_weights->params.scale;
  recurrent_to_input_weight_scale = recurrent_to_input_weights->params.scale;

  output_state_scale = output_state->params.scale;

  input_to_forget_weight_scale = input_to_forget_weights->params.scale;
  input_to_cell_weight_scale = input_to_cell_weights->params.scale;
  input_to_output_weight_scale = input_to_output_weights->params.scale;
  recurrent_to_forget_weight_scale = recurrent_to_forget_weights->params.scale;
  recurrent_to_cell_weight_scale = recurrent_to_cell_weights->params.scale;
  recurrent_to_output_weight_scale = recurrent_to_output_weights->params.scale;

  // Check cell state (already used above)
  TF_LITE_ENSURE(context, CheckedLog2(cell_state->params.scale, &cell_scale));
  TF_LITE_ENSURE(context, cell_scale <= -9);

  op_data->params_cmsis_nn.cell_state_shift = cell_scale;
  input_scale = input->params.scale;

  // Calculate effective scales.
  effective_input_to_input_scale =
      input_to_input_weight_scale * input_scale / intermediate_scale[0];
  effective_recurrent_to_input_scale = recurrent_to_input_weight_scale *
                                       output_state_scale /
                                       intermediate_scale[0];

  effective_input_to_forget_scale =
      input_to_forget_weight_scale * input_scale / intermediate_scale[1];
  effective_recurrent_to_forget_scale = recurrent_to_forget_weight_scale *
                                        output_state_scale /
                                        intermediate_scale[1];

  effective_input_to_cell_scale =
      input_to_cell_weight_scale * input_scale / intermediate_scale[2];
  effective_recurrent_to_cell_scale = recurrent_to_cell_weight_scale *
                                      output_state_scale /
                                      intermediate_scale[2];

  effective_input_to_output_scale =
      input_to_output_weight_scale * input_scale / intermediate_scale[3];
  effective_recurrent_to_output_scale = recurrent_to_output_weight_scale *
                                        output_state_scale /
                                        intermediate_scale[3];

  effective_hidden_scale =
      std::pow(2.0f, -15.0f) / intermediate_scale[4] * std::pow(2.0f, -15.0f);

  // Decompose scales.
  int shift_output;
  QuantizeMultiplier(
      static_cast<double>(effective_input_to_input_scale),
      &op_data->params_cmsis_nn.input_to_input_scaling.multiplier,
      &shift_output);
  op_data->params_cmsis_nn.input_to_input_scaling.shift =
      static_cast<int32_t>(shift_output);

  QuantizeMultiplier(
      static_cast<double>(effective_recurrent_to_input_scale),
      &op_data->params_cmsis_nn.recurrent_to_input_scaling.multiplier,
      &shift_output);
  op_data->params_cmsis_nn.recurrent_to_input_scaling.shift =
      static_cast<int32_t>(shift_output);
  QuantizeMultiplier(static_cast<double>(effective_cell_to_input_scale),
                     &op_data->params_cmsis_nn.cell_to_input_scaling.multiplier,
                     &shift_output);
  op_data->params_cmsis_nn.cell_to_input_scaling.shift =
      static_cast<int32_t>(shift_output);
  QuantizeMultiplier(
      static_cast<double>(effective_input_to_forget_scale),
      &op_data->params_cmsis_nn.input_to_forget_scaling.multiplier,
      &shift_output);
  op_data->params_cmsis_nn.input_to_forget_scaling.shift =
      static_cast<int32_t>(shift_output);
  QuantizeMultiplier(
      static_cast<double>(effective_recurrent_to_forget_scale),
      &op_data->params_cmsis_nn.recurrent_to_forget_scaling.multiplier,
      &shift_output);
  op_data->params_cmsis_nn.recurrent_to_forget_scaling.shift =
      static_cast<int32_t>(shift_output);
  QuantizeMultiplier(
      static_cast<double>(effective_cell_to_forget_scale),
      &op_data->params_cmsis_nn.cell_to_forget_scaling.multiplier,
      &shift_output);
  // ok
  op_data->params_cmsis_nn.cell_to_forget_scaling.shift =
      static_cast<int32_t>(shift_output);
  QuantizeMultiplier(static_cast<double>(effective_input_to_cell_scale),
                     &op_data->params_cmsis_nn.input_to_cell_scaling.multiplier,
                     &shift_output);
  op_data->params_cmsis_nn.input_to_cell_scaling.shift =
      static_cast<int32_t>(shift_output);
  QuantizeMultiplier(
      static_cast<double>(effective_recurrent_to_cell_scale),
      &op_data->params_cmsis_nn.recurrent_to_cell_scaling.multiplier,
      &shift_output);
  op_data->params_cmsis_nn.recurrent_to_cell_scaling.shift =
      static_cast<int32_t>(shift_output);
  QuantizeMultiplier(
      static_cast<double>(effective_input_to_output_scale),
      &op_data->params_cmsis_nn.input_to_output_scaling.multiplier,
      &shift_output);
  op_data->params_cmsis_nn.input_to_output_scaling.shift =
      static_cast<int32_t>(shift_output);
  QuantizeMultiplier(
      static_cast<double>(effective_recurrent_to_output_scale),
      &op_data->params_cmsis_nn.recurrent_to_output_scaling.multiplier,
      &shift_output);
  op_data->params_cmsis_nn.recurrent_to_output_scaling.shift =
      static_cast<int32_t>(shift_output);
  QuantizeMultiplier(
      static_cast<double>(effective_cell_to_output_scale),
      &op_data->params_cmsis_nn.cell_to_output_scaling.multiplier,
      &shift_output);
  op_data->params_cmsis_nn.cell_to_output_scaling.shift =
      static_cast<int32_t>(shift_output);

  op_data->params_cmsis_nn.projection_scaling.shift =
      static_cast<int32_t>(shift_output);

  QuantizeMultiplier(static_cast<double>(effective_hidden_scale),
                     &op_data->params_cmsis_nn.hidden_scaling.multiplier,
                     &shift_output);
  op_data->params_cmsis_nn.hidden_scaling.shift =
      static_cast<int32_t>(shift_output);

  op_data->params_cmsis_nn.hidden_offset = intermediate_zp[4];

  op_data->params_cmsis_nn.activation.min = std::numeric_limits<int16_t>::min();
  op_data->params_cmsis_nn.activation.max = std::numeric_limits<int16_t>::max();

  return kTfLiteOk;
}

template <typename CellType>
TfLiteStatus CMSIS_NN_EvalInteger8x8_16Lstm(
    const OpData& op_data, const LSTMKernelContents& kernel_content,
    const LSTMBuffers<CellType>& buffers) {
  const OpDataLSTM& op_data_lstm = op_data.params_ref;
  const TfLiteEvalTensor* input =
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor);
  const TfLiteEvalTensor* input_gate_bias =
      kernel_content.GetInternalTensor(tflite::kLstmInputGateBiasTensor);
  const TfLiteEvalTensor* forget_gate_bias =
      kernel_content.GetInternalTensor(tflite::kLstmForgetGateBiasTensor);
  const TfLiteEvalTensor* cell_gate_bias =
      kernel_content.GetInternalTensor(tflite::kLstmCellGateBiasTensor);
  const TfLiteEvalTensor* output_gate_bias =
      kernel_content.GetInternalTensor(tflite::kLstmOutputGateBiasTensor);
  const TfLiteEvalTensor* input_to_output_weights =
      kernel_content.GetInternalTensor(tflite::kLstmInputToOutputWeightsTensor);
  const TfLiteEvalTensor* recurrent_to_output_weights =
      kernel_content.GetInternalTensor(
          tflite::kLstmRecurrentToOutputWeightsTensor);
  const TfLiteEvalTensor* input_to_input_weights =
      kernel_content.GetInternalTensor(tflite::kLstmInputToInputWeightsTensor);
  const TfLiteEvalTensor* input_to_forget_weights =
      kernel_content.GetInternalTensor(tflite::kLstmInputToForgetWeightsTensor);
  const TfLiteEvalTensor* input_to_cell_weights =
      kernel_content.GetInternalTensor(tflite::kLstmInputToCellWeightsTensor);
  const TfLiteEvalTensor* recurrent_to_input_weights =
      kernel_content.GetInternalTensor(
          tflite::kLstmRecurrentToInputWeightsTensor);
  const TfLiteEvalTensor* recurrent_to_forget_weights =
      kernel_content.GetInternalTensor(
          tflite::kLstmRecurrentToForgetWeightsTensor);
  const TfLiteEvalTensor* recurrent_to_cell_weights =
      kernel_content.GetInternalTensor(
          tflite::kLstmRecurrentToCellWeightsTensor);
  const TfLiteEvalTensor* cell_to_input_weights =
      kernel_content.GetInternalTensor(tflite::kLstmCellToInputWeightsTensor);
  const TfLiteEvalTensor* cell_to_forget_weights =
      kernel_content.GetInternalTensor(tflite::kLstmCellToForgetWeightsTensor);
  const TfLiteEvalTensor* cell_to_output_weights =
      kernel_content.GetInternalTensor(tflite::kLstmCellToOutputWeightsTensor);
  const TfLiteEvalTensor* cell_state =
      kernel_content.GetInternalTensor(tflite::kLstmCellStateTensor);
  const TfLiteEvalTensor* output_state =
      kernel_content.GetInternalTensor(tflite::kLstmOutputStateTensor);
  const TfLiteEvalTensor* output = kernel_content.output_tensor;

  TFLITE_DCHECK(input->dims->size >= 2 && input->dims->size <= 3);

  cmsis_nn_lstm_context scratch_buffers;
  scratch_buffers.input_gate = reinterpret_cast<int16_t*>(buffers.buffer0);
  scratch_buffers.forget_gate = reinterpret_cast<int16_t*>(buffers.buffer1);
  scratch_buffers.cell_gate = reinterpret_cast<int16_t*>(buffers.buffer2);
  scratch_buffers.output_gate = reinterpret_cast<int16_t*>(buffers.buffer3);

  cmsis_nn_lstm_params cmsis_lstm_params = op_data.params_cmsis_nn;
  cmsis_lstm_params.time_major = op_data_lstm.size_info.time_major;
  cmsis_lstm_params.clip.cell =
      op_data_lstm.cell_state_info.quantized_cell_clip;

  cmsis_lstm_params.input_gate_bias = const_cast<int32_t*>(
      tflite::micro::GetOptionalTensorData<int32_t>(input_gate_bias));
  cmsis_lstm_params.forget_gate_bias = const_cast<int32_t*>(
      tflite::micro::GetOptionalTensorData<int32_t>(forget_gate_bias));
  cmsis_lstm_params.cell_gate_bias = const_cast<int32_t*>(
      tflite::micro::GetOptionalTensorData<int32_t>(cell_gate_bias));
  cmsis_lstm_params.output_gate_bias = const_cast<int32_t*>(
      tflite::micro::GetOptionalTensorData<int32_t>(output_gate_bias));

  const bool time_major = op_data_lstm.size_info.time_major;
  const int n_input = input->dims->data[input->dims->size - 1];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  int max_time, n_batch;
  if (input->dims->size == 2) {
    max_time = 1;
    n_batch = input->dims->data[0];
  } else {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  }

  cmsis_nn_lstm_dims lstm_dims;
  lstm_dims.num_inputs = n_input;
  lstm_dims.num_outputs = n_output;
  lstm_dims.num_batches = n_batch;
  lstm_dims.max_time = max_time;

  arm_lstm_unidirectional_s16_s8(
      &scratch_buffers,
      const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input)),
      &lstm_dims,
      const_cast<int8_t*>(
          tflite::micro::GetOptionalTensorData<int8_t>(input_to_input_weights)),
      const_cast<int8_t*>(tflite::micro::GetOptionalTensorData<int8_t>(
          input_to_forget_weights)),
      const_cast<int8_t*>(
          tflite::micro::GetOptionalTensorData<int8_t>(input_to_cell_weights)),
      const_cast<int8_t*>(tflite::micro::GetOptionalTensorData<int8_t>(
          input_to_output_weights)),
      const_cast<int8_t*>(tflite::micro::GetOptionalTensorData<int8_t>(
          recurrent_to_input_weights)),
      const_cast<int8_t*>(tflite::micro::GetOptionalTensorData<int8_t>(
          recurrent_to_forget_weights)),
      const_cast<int8_t*>(tflite::micro::GetOptionalTensorData<int8_t>(
          recurrent_to_cell_weights)),
      const_cast<int8_t*>(tflite::micro::GetOptionalTensorData<int8_t>(
          recurrent_to_output_weights)),
      const_cast<int16_t*>(
          tflite::micro::GetOptionalTensorData<int16_t>(cell_to_input_weights)),
      const_cast<int16_t*>(tflite::micro::GetOptionalTensorData<int16_t>(
          cell_to_forget_weights)),
      const_cast<int16_t*>(tflite::micro::GetOptionalTensorData<int16_t>(
          cell_to_output_weights)),
      nullptr, &cmsis_lstm_params,
      const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(output_state)),
      const_cast<int16_t*>(tflite::micro::GetTensorData<int16_t>(cell_state)),
      const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(output)));

  return kTfLiteOk;
}

/*Kernel functions*/

void* UnidirectionalSequenceLstmInit(TfLiteContext* context, const char* buffer,
                                     size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus UnidirectionalSequenceLstmPrepare(TfLiteContext* context,
                                               TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 24);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  TFLITE_DCHECK(node->user_data != nullptr);

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  OpDataLSTM* op_data_lstm = &op_data->params_ref;

  const auto* builtin_data =
      static_cast<TfLiteUnidirectionalSequenceLSTMParams*>(node->builtin_data);
  // All TempTfLiteTensors will be deallocated through the destructor.
  LstmTensors lstm_tensors(context, node);
  TF_LITE_ENSURE_OK(context, lstm_tensors.ValidateTensorStatus(context));

  op_data_lstm->cell_gate_nonlinear_type = builtin_data->activation;
  op_data_lstm->size_info =
      CreateLstmSizeInfo(builtin_data->time_major,
                         lstm_tensors.GetInternalTensor(kLstmInputTensor)->dims,
                         lstm_tensors.HiddenStateTensor()->dims);

  const TfLiteTensor* input = lstm_tensors.GetInternalTensor(kLstmInputTensor);
  const auto activation_type = input->type;

  if (kTfLiteInt8 == activation_type) {
    TF_LITE_ENSURE_STATUS(
        CalculateOpData(context, node, lstm_tensors, op_data));
  }

  TF_LITE_ENSURE_OK(context, ValidateTensorSize(context, lstm_tensors,
                                                op_data_lstm->size_info));

  // Create cell state information and gate parameters (Fully Connected and Mul)
  auto cell_state_type =
      lstm_tensors.GetInternalTensor(kLstmCellStateTensor)->type;
  if (cell_state_type == kTfLiteFloat32) {
    op_data_lstm->cell_state_info =
        CreateLstmCellStateInfoFloat(builtin_data->cell_clip);
    TF_LITE_ENSURE_OK(context, PrepareGateParametersFloat(context, lstm_tensors,
                                                          op_data_lstm));
  } else if (cell_state_type == kTfLiteInt16) {
    op_data_lstm->cell_state_info = CreateLstmCellStateInfo(
        lstm_tensors.CellStateTensor()->params.scale, builtin_data->cell_clip);
    TF_LITE_ENSURE_OK(context, PrepareGateParametersInteger(
                                   context, lstm_tensors, op_data_lstm));
  } else {
    MicroPrintf(
        "Cell state type %s (%d) not supported. The quantized Unidirectional "
        "Sequence LSTM Op only support int16 cell state",
        TfLiteTypeGetName(cell_state_type), cell_state_type);
    return kTfLiteError;
  }
  // request buffers (four buffers)
  for (size_t i = 0; i < 4; i++) {
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context,
                                   op_data_lstm->size_info.batch_size *
                                       op_data_lstm->size_info.state_dimension *
                                       TfLiteTypeGetSize(cell_state_type),
                                   &(op_data_lstm->buffer_indices[i])));
  }

  return kTfLiteOk;
}

TfLiteStatus UnidirectionalSequenceLstmEval(TfLiteContext* context,
                                            TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& op_data = *reinterpret_cast<const OpData*>(node->user_data);
  const OpDataLSTM& op_data_lstm = op_data.params_ref;

  auto kernel_content = CreateLSTMKernelContent(context, node);

  const auto activation_type =
      kernel_content.internal_tensors[kLstmInputTensor]->type;
  const auto weight_type =
      kernel_content.internal_tensors[kLstmInputToInputWeightsTensor]->type;

  switch (activation_type) {
    case kTfLiteFloat32: {
      LSTMBuffers<float> buffers =
          CreateLSTMBuffers<float>(context, op_data_lstm.buffer_indices);
      EvalLstm<float, float, float, float>(op_data_lstm, kernel_content,
                                           buffers);
      break;
    }
    case kTfLiteInt8: {
      switch (weight_type) {
        case kTfLiteInt8: {
          // 8(activation)x8(weight)->16(cell) LSTM with 32 bits bias
          LSTMBuffers<int16_t> buffers =
              CreateLSTMBuffers<int16_t>(context, op_data_lstm.buffer_indices);
          return CMSIS_NN_EvalInteger8x8_16Lstm<int16_t>(
              op_data, kernel_content, buffers);
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(weight_type), activation_type);
          return kTfLiteError;
        }
      }
      break;
    }
    case kTfLiteInt16: {
      switch (weight_type) {
        case kTfLiteInt8: {
          // 16(activation)x8(weight)->16(cell) LSTM with 64 bits bias
          LSTMBuffers<int16_t> buffers =
              CreateLSTMBuffers<int16_t>(context, op_data_lstm.buffer_indices);
          EvalLstm<int16_t, int8_t, int16_t, int64_t>(op_data_lstm,
                                                      kernel_content, buffers);
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(weight_type), weight_type);
          return kTfLiteError;
        }
      }
      break;
    }
    default: {
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(activation_type), activation_type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus UnidirectionalSequenceLstmEvalInt8(TfLiteContext* context,
                                                TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& op_data = *reinterpret_cast<const OpData*>(node->user_data);
  const OpDataLSTM& op_data_lstm = op_data.params_ref;
  auto kernel_content = CreateLSTMKernelContent(context, node);
  const auto activation_type =
      kernel_content.internal_tensors[kLstmInputTensor]->type;
  const auto weight_type =
      kernel_content.internal_tensors[kLstmInputToInputWeightsTensor]->type;

  TFLITE_DCHECK(weight_type == kTfLiteInt16 &&
                "Only int16 filter type supported.");

  if (activation_type == kTfLiteInt8) {
    LSTMBuffers<int16_t> buffers =
        CreateLSTMBuffers<int16_t>(context, op_data_lstm.buffer_indices);

    return CMSIS_NN_EvalInteger8x8_16Lstm<int16_t>(op_data, kernel_content,
                                                   buffers);
  } else {
    MicroPrintf("Input type %s (%d) not supported.",
                TfLiteTypeGetName(activation_type), activation_type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM() {
  return tflite::micro::RegisterOp(UnidirectionalSequenceLstmInit,
                                   UnidirectionalSequenceLstmPrepare,
                                   UnidirectionalSequenceLstmEval);
}

TfLiteRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT8() {
  return tflite::micro::RegisterOp(UnidirectionalSequenceLstmInit,
                                   UnidirectionalSequenceLstmPrepare,
                                   UnidirectionalSequenceLstmEvalInt8);
}

}  // namespace tflite
