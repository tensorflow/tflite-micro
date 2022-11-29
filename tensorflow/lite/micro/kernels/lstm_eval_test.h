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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_TEST_COMMOM_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_TEST_COMMOM_H_
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

// A function that converts floating point gate parameters to the
// corresponding quantized version
template <typename WeightType, typename BiasType, int input_dimension,
          int state_dimension>
GateParameters<WeightType, BiasType, input_dimension, state_dimension>
CreateQuantizedGateParameters(
    const GateParameters<float, float, input_dimension, state_dimension>&
        gate_parameters,
    const TensorQuantizationParameters& input_quantization_params,
    const TensorQuantizationParameters& output_quantization_params,
    const GateQuantizationParameters& gate_quantization_params) {
  GateParameters<WeightType, BiasType, input_dimension, state_dimension>
      quantized_gate_params;
  tflite::SymmetricQuantize(gate_parameters.activation_weight,
                            quantized_gate_params.activation_weight,
                            state_dimension * input_dimension,
                            gate_quantization_params.activation_weight.scale);
  tflite::SymmetricQuantize(gate_parameters.recurrent_weight,
                            quantized_gate_params.recurrent_weight,
                            state_dimension * state_dimension,
                            gate_quantization_params.recurrent_weight.scale);
  tflite::SymmetricQuantize(gate_parameters.fused_bias,
                            quantized_gate_params.fused_bias, state_dimension,
                            gate_quantization_params.bias.scale);

  // Copy the bias values to prepare zero_point folded bias precomputation. bias
  // has same scale as input_scale*input_weight_scale)
  std::memcpy(quantized_gate_params.activation_zp_folded_bias,
              quantized_gate_params.fused_bias,
              state_dimension * sizeof(BiasType));
  // Pre-calculate bias - zero_point * weight (a constant).
  tflite::tensor_utils::MatrixScalarMultiplyAccumulate(
      quantized_gate_params.activation_weight,
      -1 * input_quantization_params.zero_point, state_dimension,
      input_dimension, quantized_gate_params.activation_zp_folded_bias);

  // Initialize the folded bias to zeros for accumulation
  for (size_t i = 0; i < state_dimension; i++) {
    quantized_gate_params.recurrent_zp_folded_bias[i] = 0;
  }
  // Calculate : -zero_point * weight since it is a constant
  tflite::tensor_utils::MatrixScalarMultiplyAccumulate(
      quantized_gate_params.recurrent_weight,
      -1 * output_quantization_params.zero_point, state_dimension,
      state_dimension, quantized_gate_params.recurrent_zp_folded_bias);

  return quantized_gate_params;
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
ModelContents<int8_t, int8_t, int32_t, int16_t, batch_size, time_steps,
              input_dimension, state_dimension>
CreateInt8ModelContents(
    const ModelQuantizationParameters& quantization_settings,
    ModelContents<float, float, float, float, batch_size, time_steps,
                  input_dimension, state_dimension>& float_model_contents) {
  auto quantized_forget_gate_params =
      CreateQuantizedGateParameters<int8_t, int32_t, input_dimension,
                                    state_dimension>(
          float_model_contents.ForgetGateParams(),
          quantization_settings.input_quantization_parameters,
          quantization_settings.output_quantization_parameters,
          quantization_settings.forget_gate_quantization_parameters);
  auto quantized_input_gate_params =
      CreateQuantizedGateParameters<int8_t, int32_t, input_dimension,
                                    state_dimension>(
          float_model_contents.InputGateParams(),
          quantization_settings.input_quantization_parameters,
          quantization_settings.output_quantization_parameters,
          quantization_settings.forget_gate_quantization_parameters);
  auto quantized_cell_gate_params =
      CreateQuantizedGateParameters<int8_t, int32_t, input_dimension,
                                    state_dimension>(
          float_model_contents.CellGateParams(),
          quantization_settings.input_quantization_parameters,
          quantization_settings.output_quantization_parameters,
          quantization_settings.forget_gate_quantization_parameters);
  auto quantized_output_gate_params =
      CreateQuantizedGateParameters<int8_t, int32_t, input_dimension,
                                    state_dimension>(
          float_model_contents.OutputGateParams(),
          quantization_settings.input_quantization_parameters,
          quantization_settings.output_quantization_parameters,
          quantization_settings.forget_gate_quantization_parameters);
  return ModelContents<int8_t, int8_t, int32_t, int16_t, batch_size, time_steps,
                       input_dimension, state_dimension>(
      quantized_forget_gate_params, quantized_input_gate_params,
      quantized_cell_gate_params, quantized_output_gate_params);
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
IntegerLstmParameter CreateIntegerParameter(
    const TfLiteLSTMParams& general_model_settings,
    const ModelQuantizationParameters& quantization_settings,
    ModelContents<int8_t, int8_t, int32_t, int16_t, batch_size, time_steps,
                  input_dimension, state_dimension>& quantized_model_contents) {
  IntegerLstmParameter evaluation_params;
  double effective_scale;
  // TODO(b/260006407): QuantizeMultiplier takes int as the output shift
  // type, but the shift type is stored as int32_t inside the
  // IntegerLstmParameter. Hexagon compilation requires the exact match of the
  // two. Considering make shift type to be int inside the
  // IntegerLstmParameter.
  int buffer_shift_output;
  // TODO(b/253466487): Considering refactoring IntegerLstmParameter to
  // distribute the calculation of gate quantization parameters (e.g.,
  // effective scale) to gate level. Forget Gate
  effective_scale = quantization_settings.input_quantization_parameters.scale *
                    quantization_settings.forget_gate_quantization_parameters
                        .activation_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_input_to_forget_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_input_to_forget_scale_b = buffer_shift_output;
  effective_scale = quantization_settings.output_quantization_parameters.scale *
                    quantization_settings.forget_gate_quantization_parameters
                        .recurrent_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_recurrent_to_forget_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_recurrent_to_forget_scale_b = buffer_shift_output;
  // Set effective bias
  evaluation_params.input_to_forget_effective_bias =
      quantized_model_contents.ForgetGateParams().activation_zp_folded_bias;
  evaluation_params.recurrent_to_forget_effective_bias =
      quantized_model_contents.ForgetGateParams().recurrent_zp_folded_bias;

  // input gate
  effective_scale = quantization_settings.input_quantization_parameters.scale *
                    quantization_settings.input_gate_quantization_parameters
                        .activation_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_input_to_input_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_input_to_input_scale_b = buffer_shift_output;
  effective_scale = quantization_settings.output_quantization_parameters.scale *
                    quantization_settings.input_gate_quantization_parameters
                        .recurrent_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_recurrent_to_input_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_recurrent_to_input_scale_b = buffer_shift_output;
  // Set effective bias
  evaluation_params.input_to_input_effective_bias =
      quantized_model_contents.InputGateParams().activation_zp_folded_bias;
  evaluation_params.recurrent_to_input_effective_bias =
      quantized_model_contents.InputGateParams().recurrent_zp_folded_bias;

  // cell gate
  effective_scale = quantization_settings.input_quantization_parameters.scale *
                    quantization_settings.cell_gate_quantization_parameters
                        .activation_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_input_to_cell_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_input_to_cell_scale_b = buffer_shift_output;
  effective_scale = quantization_settings.output_quantization_parameters.scale *
                    quantization_settings.cell_gate_quantization_parameters
                        .recurrent_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_recurrent_to_cell_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_recurrent_to_cell_scale_b = buffer_shift_output;
  // Set effective bias
  evaluation_params.input_to_cell_effective_bias =
      quantized_model_contents.CellGateParams().activation_zp_folded_bias;
  evaluation_params.recurrent_to_cell_effective_bias =
      quantized_model_contents.CellGateParams().recurrent_zp_folded_bias;

  // output gate
  effective_scale = quantization_settings.input_quantization_parameters.scale *
                    quantization_settings.output_gate_quantization_parameters
                        .activation_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_input_to_output_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_input_to_output_scale_b = buffer_shift_output;
  effective_scale = quantization_settings.output_quantization_parameters.scale *
                    quantization_settings.output_gate_quantization_parameters
                        .recurrent_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_recurrent_to_output_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_recurrent_to_output_scale_b = buffer_shift_output;
  // Set effective bias
  evaluation_params.input_to_output_effective_bias =
      quantized_model_contents.OutputGateParams().activation_zp_folded_bias;
  evaluation_params.recurrent_to_output_effective_bias =
      quantized_model_contents.OutputGateParams().recurrent_zp_folded_bias;

  // hidden state (no projection, output is the hidden state)
  effective_scale = quantization_settings.nonlinear_activation_output_scale *
                    quantization_settings.nonlinear_activation_output_scale /
                    quantization_settings.hidden_quantization_parameters.scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_hidden_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_hidden_scale_b = buffer_shift_output;
  evaluation_params.hidden_zp =
      quantization_settings.hidden_quantization_parameters.zero_point;

  // cell state. Note, cell_scale is actually not a scale. 2^-cell_scale is
  // the true scale for cell
  int buffer_cell_scale;
  tflite::CheckedLog2(quantization_settings.cell_quantization_parameters.scale,
                      &buffer_cell_scale);
  evaluation_params.cell_scale = buffer_cell_scale;

  evaluation_params.quantized_cell_clip = static_cast<int16_t>(std::min(
      std::max(static_cast<double>(general_model_settings.cell_clip) /
                   quantization_settings.cell_quantization_parameters.scale,
               -32768.0),
      32767.0));
  return evaluation_params;
}

// Create a 2x2 quantized 8x8->16 model content
// batch_size = 2; time_steps = 3; input_dimension = 2; state_dimension = 2
ModelContents<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
Create2x3x2X2Int8ModelContents(
    const ModelQuantizationParameters& quantization_settings);

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_TEST_COMMOM_H_
