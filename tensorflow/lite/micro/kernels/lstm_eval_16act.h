/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_16ACT_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_16ACT_H_

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
// Since LSTM includes multiple intermediate stages, introducing the internal
// namespace to expose them for testing
namespace lstm_internal {

// struct GateParameters {
//   FullyConnectedParams input_fc_params;
//   FullyConnectedParams recurrent_fc_params;
// };

// struct InterGateParameters {
//   ArithmeticParams forget_cell_mul_params;
//   ArithmeticParams input_mul_params;
//   ArithmeticParams output_mul_params;
// };

// struct ScratchBuffers {
//   int scratch_index[4];  // four buffers
// };

// struct OpDataLSTM {
//   GateParameters forget_gate_parameters;
//   GateParameters input_gate_parameters;
//   GateParameters cell_gate_parameters;
//   GateParameters output_gate_parameters;
//   InterGateParameters inter_gate_parameters;
//   ScratchBuffers buffers;  // TFLM only
// };

// Calculates a single LSTM gate.
// Implements the following formula:
//   gate = activate(FC(input) + FC(recurrent))
// Activation is sigmoid except for the "cell" gate (configurable, usually tanh)
template <typename ActivationType, typename WeightType, typename CellType,
          typename BiasType>
void CalculateLstmGateInteger(
    // Input FC
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* input_weight,
    const TfLiteEvalTensor* input_bias,
    const FullyConnectedParams& input_fc_params,
    // Recurrent FC
    const TfLiteEvalTensor* recurrent, const TfLiteEvalTensor* recurrent_weight,
    const TfLiteEvalTensor* recurrent_bias,
    const FullyConnectedParams& recurrent_fc_params,
    // Output
    CellType* gate_output,
    // Scratch arrays
    CellType* fc_output_buffer, const TfLiteFusedActivation activation) {
  // gate output has the same size of the recurrent
  const auto gate_output_shape = tflite::micro::GetTensorShape(recurrent);
  // Input FC
  tflite::reference_integer_ops::FullyConnectedGeneral<
      ActivationType, CellType, WeightType, BiasType, int64_t>(
      input_fc_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<ActivationType>(input),
      micro::GetTensorShape(input_weight),
      tflite::micro::GetTensorData<WeightType>(input_weight),
      tflite::micro::GetTensorShape(input_bias),
      tflite::micro::GetOptionalTensorData<BiasType>(input_bias),
      gate_output_shape, gate_output);

  // Recurrent FC
  tflite::reference_integer_ops::FullyConnectedGeneral<
      ActivationType, CellType, WeightType, BiasType, int32_t>(
      recurrent_fc_params, tflite::micro::GetTensorShape(recurrent),
      tflite::micro::GetTensorData<ActivationType>(recurrent),
      tflite::micro::GetTensorShape(recurrent_weight),
      tflite::micro::GetTensorData<WeightType>(recurrent_weight),
      tflite::micro::GetTensorShape(recurrent_bias),
      tflite::micro::GetOptionalTensorData<BiasType>(recurrent_bias),
      gate_output_shape, fc_output_buffer);

  tflite::tensor_utils::CwiseAdd(gate_output, fc_output_buffer,
                                 /*n_batch=*/gate_output_shape.DimsData()[0],
                                 /*n_state=*/gate_output_shape.DimsData()[1],
                                 gate_output);

  // Apply activation
  switch (activation) {
    case kTfLiteActSigmoid:
      reference_integer_ops::Logistic(
          0 /*data->input_multiplier*/, 0 /*data->input_left_shift */,
          gate_output_shape.FlatSize() /*NumElements(input->dims)*/,
          gate_output /* tflite::micro::GetTensorData<int16_t>(input) */,
          gate_output /*tflite::micro::GetTensorData<int16_t>(output) */);

      break;
    case kTfLiteActTanh: {
      reference_integer_ops::Tanh(0, 0, gate_output_shape, gate_output,
                                  gate_output_shape, gate_output);
    } break;
    default:
      // Only Sigmoid or Tanh is used.
      TFLITE_ASSERT_FALSE;
  }
}

template <typename CellType>
void UpdateLstmCellInteger(TfLiteEvalTensor* cell_state,
                           // Gate outputs
                           CellType* forget_gate_output,
                           const CellType* input_gate_output,
                           const CellType* cell_gate_output,
                           // Mul parameters
                           const ArithmeticParams& forget_cell_mul_params,
                           const ArithmeticParams& input_mul_params,
                           CellType* buffer, CellType clip) {
  auto cell_state_shape = tflite::micro::GetTensorShape(cell_state);
  // Forget Gate x Cell State
  tflite::reference_integer_ops::MulElementwise(
      cell_state_shape.FlatSize(), forget_cell_mul_params, forget_gate_output,
      tflite::micro::GetTensorData<CellType>(cell_state),
      tflite::micro::GetTensorData<CellType>(cell_state));

  // Input Gate x Cell Gate
  tflite::reference_integer_ops::MulElementwise(
      cell_state_shape.FlatSize(), input_mul_params, input_gate_output,
      cell_gate_output, buffer);

  // Update the cell state
  tflite::tensor_utils::CwiseAdd(
      tflite::micro::GetTensorData<CellType>(cell_state), buffer,
      /*n_batch=*/cell_state_shape.DimsData()[0],
      /*n_state=*/cell_state_shape.DimsData()[1],
      tflite::micro::GetTensorData<CellType>(cell_state));

  if (clip > 0) {
    tflite::tensor_utils::CwiseClipping(
        tflite::micro::GetTensorData<CellType>(cell_state),
        cell_state_shape.FlatSize(), clip);
  }
}

template <typename CellType, typename ActivationType>
void UpdateLstmHiddenInteger(TfLiteEvalTensor* cell_state,
                             TfLiteEvalTensor* hidden_state,
                             const CellType* output_gate_output,
                             const ArithmeticParams& mul_params,
                             int32_t cell_state_scale, CellType* buffer) {
  auto cell_state_shape = tflite::micro::GetTensorShape(cell_state);
  CellType* cell_state_data =
      tflite::micro::GetTensorData<CellType>(cell_state);
  // Tanh(cell_state)
  {
    int32_t tanh_input_left_shift = (15 + cell_state_scale) - 3;
    if (tanh_input_left_shift < 0) /* handling negative shift value */
    {
      int32_t i;
      tanh_input_left_shift = -tanh_input_left_shift;
      for (i = 0; i < cell_state_shape.FlatSize(); i++) {
        cell_state_data[i] = cell_state_data[i] >> tanh_input_left_shift;
      }
      tanh_input_left_shift = 0;
    }
    reference_integer_ops::Tanh(0, tanh_input_left_shift, cell_state_shape,
                                cell_state_data, cell_state_shape, buffer);
  }
  // Update the hidden state
  tflite::reference_integer_ops::MulElementwiseGeneral(
      cell_state_shape.FlatSize(), mul_params, buffer, output_gate_output,
      tflite::micro::GetTensorData<ActivationType>(hidden_state));
}

}  // namespace lstm_internal
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_16ACT_H_