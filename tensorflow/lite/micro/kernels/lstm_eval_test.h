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
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
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

// Create int8 (activation) x int8 (weight) -> int16 (cell) model from the float
// model contents and quantization settings
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
          quantization_settings.input_gate_quantization_parameters);
  auto quantized_cell_gate_params =
      CreateQuantizedGateParameters<int8_t, int32_t, input_dimension,
                                    state_dimension>(
          float_model_contents.CellGateParams(),
          quantization_settings.input_quantization_parameters,
          quantization_settings.output_quantization_parameters,
          quantization_settings.cell_gate_quantization_parameters);
  auto quantized_output_gate_params =
      CreateQuantizedGateParameters<int8_t, int32_t, input_dimension,
                                    state_dimension>(
          float_model_contents.OutputGateParams(),
          quantization_settings.input_quantization_parameters,
          quantization_settings.output_quantization_parameters,
          quantization_settings.output_gate_quantization_parameters);
  ModelContents<int8_t, int8_t, int32_t, int16_t, batch_size, time_steps,
                input_dimension, state_dimension>
      quantized_model_content(
          quantized_forget_gate_params, quantized_input_gate_params,
          quantized_cell_gate_params, quantized_output_gate_params);

  // Quantize the  floating point input
  int8_t quantized_input[batch_size * input_dimension * time_steps] = {};
  Quantize(float_model_contents.GetInput(), quantized_input,
           batch_size * input_dimension * time_steps,
           quantization_settings.input_quantization_parameters.scale,
           quantization_settings.input_quantization_parameters.zero_point);
  quantized_model_content.SetInputTensorData(quantized_input);
  // Quantize the  floating point hidden state
  int8_t quantized_hidden_state[batch_size * state_dimension] = {};
  Quantize(float_model_contents.GetHiddenState(), quantized_hidden_state,
           batch_size * state_dimension,
           quantization_settings.hidden_quantization_parameters.scale,
           quantization_settings.hidden_quantization_parameters.zero_point);
  quantized_model_content.SetHiddenStateTensorData(quantized_hidden_state);
  // Quantize the floating point cell state
  int16_t quantized_cell_state[batch_size * state_dimension] = {};
  Quantize(float_model_contents.GetCellState(), quantized_cell_state,
           batch_size * state_dimension,
           quantization_settings.cell_quantization_parameters.scale,
           quantization_settings.cell_quantization_parameters.zero_point);
  quantized_model_content.SetCellStateTensorData(quantized_cell_state);
  return quantized_model_content;
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
IntegerLstmParameter CreateIntegerParameter(
    const TfLiteLSTMParams& general_model_settings,
    const ModelQuantizationParameters& quantization_settings,
    const ModelContents<int8_t, int8_t, int32_t, int16_t, batch_size,
                        time_steps, input_dimension, state_dimension>&
        quantized_model_contents) {
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
  evaluation_params.input_to_forget_effective_bias = const_cast<int32_t*>(
      quantized_model_contents.ForgetGateParams().activation_zp_folded_bias);
  evaluation_params.recurrent_to_forget_effective_bias = const_cast<int32_t*>(
      quantized_model_contents.ForgetGateParams().recurrent_zp_folded_bias);

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
  evaluation_params.input_to_input_effective_bias = const_cast<int32_t*>(
      quantized_model_contents.InputGateParams().activation_zp_folded_bias);
  evaluation_params.recurrent_to_input_effective_bias = const_cast<int32_t*>(
      quantized_model_contents.InputGateParams().recurrent_zp_folded_bias);

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
  evaluation_params.input_to_cell_effective_bias = const_cast<int32_t*>(
      quantized_model_contents.CellGateParams().activation_zp_folded_bias);
  evaluation_params.recurrent_to_cell_effective_bias = const_cast<int32_t*>(
      quantized_model_contents.CellGateParams().recurrent_zp_folded_bias);

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
  evaluation_params.input_to_output_effective_bias = const_cast<int32_t*>(
      quantized_model_contents.OutputGateParams().activation_zp_folded_bias);
  evaluation_params.recurrent_to_output_effective_bias = const_cast<int32_t*>(
      quantized_model_contents.OutputGateParams().recurrent_zp_folded_bias);

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

/*TEST HELPER FUNCTIONS*/
template <typename T>
void ValidateResultGoldens(const T* golden, const T* output_data,
                           const int output_len, const float tolerance) {
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

template <int batch_size, int input_dimension, int state_dimension>
void TestGateOutputFloat(const GateParameters<float, float, input_dimension,
                                              state_dimension>& gate_params,
                         TfLiteFusedActivation activation_type,
                         const float* input_data, const float* hidden_state,
                         const float* expected_vals, const float tolerance) {
  float gate_output[batch_size * state_dimension] = {};
  tflite::lstm_internal::CalculateLstmGateFloat(
      input_data, gate_params.activation_weight,
      /*aux_input=*/nullptr, /*aux_input_to_gate_weights*/ nullptr,
      hidden_state, gate_params.recurrent_weight,
      /*cell_state=*/nullptr, /*cell_to_gate_weights=*/nullptr,
      /*layer_norm_coefficients=*/nullptr, gate_params.fused_bias, batch_size,
      input_dimension, input_dimension, state_dimension, state_dimension,
      /*activation=*/activation_type, gate_output,
      /*is_input_all_zeros=*/false,
      /*is_aux_input_all_zeros=*/true);
  ValidateResultGoldens(expected_vals, gate_output,
                        batch_size * state_dimension, tolerance);
}

// TODO(b/253466487): Clean up the input parameters, which requires refactor
// IntegerLstmParameter
template <typename ActivationType, typename BiasType, typename CellType,
          int batch_size, int input_dimension, int state_dimension>
void TestGateOutputQuantized(
    const ActivationType* quantized_input,
    const ActivationType* quantized_hidden_state,
    const GateParameters<int8_t, BiasType, input_dimension, state_dimension>&
        gate_params,
    const ModelQuantizationParameters& quantization_settings,
    int32_t effective_input_to_gate_scale_a,
    int32_t effective_input_to_gate_scale_b,
    int32_t effective_recurrent_to_gate_scale_a,
    int32_t effective_recurrent_to_gate_scale_b,
    TfLiteFusedActivation nonlinear_type, const float* expected_vals,
    float tolerance) {
  CellType gate_output[batch_size * state_dimension] = {};
  BiasType scratch_buffer[batch_size * state_dimension] = {};

  tflite::lstm_internal::CalculateLstmGateInteger8x8_16(
      // Input and weights
      quantized_input, gate_params.activation_weight,
      gate_params.activation_zp_folded_bias, effective_input_to_gate_scale_a,
      effective_input_to_gate_scale_b,
      // Output state and weights
      quantized_hidden_state, gate_params.activation_weight,
      gate_params.recurrent_zp_folded_bias, effective_recurrent_to_gate_scale_a,
      effective_recurrent_to_gate_scale_b,
      // Cell state and weights
      nullptr, nullptr, 0, 0,
      // Layer normalization parameters (layer norm LSTM)
      nullptr, nullptr, 0, 0, 0,
      // Array sizes
      batch_size, input_dimension, state_dimension, state_dimension,
      nonlinear_type,
      // Output
      gate_output,
      // Parameters for performance optimizations
      // Scratch arrays
      scratch_buffer);

  float gate_output_float[batch_size * state_dimension] = {};
  Dequantize(gate_output, batch_size * state_dimension,
             quantization_settings.nonlinear_activation_output_scale, 0,
             gate_output_float);

  ValidateResultGoldens(expected_vals, gate_output_float,
                        batch_size * state_dimension, tolerance);
}

template <int batch_size, int input_dimension, int state_dimension>
void TestCellUpdateFloat(
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    const float cell_clip, const float tolerance) {
  // copy the data since it will be updated
  float cell_state[batch_size * state_dimension] = {};
  std::memcpy(cell_state, gate_output_data.cell_state,
              batch_size * state_dimension * sizeof(float));

  float forget_gate[batch_size * state_dimension] = {};
  std::memcpy(forget_gate, gate_output_data.expected_forget_gate_output,
              batch_size * state_dimension * sizeof(float));

  tflite::lstm_internal::UpdateLstmCellFloat(
      batch_size, state_dimension, cell_state,
      gate_output_data.expected_input_gate_output, forget_gate,
      gate_output_data.expected_cell_gate_output,
      /*use_cifg=*/false, cell_clip);

  ValidateResultGoldens(gate_output_data.expected_updated_cell, cell_state,
                        batch_size * state_dimension, tolerance);
}

template <typename ActivationType, typename BiasType, typename CellType,
          int batch_size, int input_dimension, int state_dimension>
void TestCellUpdateQuantized(
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    const ModelQuantizationParameters& quantization_settings,
    const int32_t cell_scale_shift, const CellType quantized_cell_clip,
    const float tolerance) {
  CellType quantized_cell_state[batch_size * state_dimension] = {};
  tflite::Quantize(
      gate_output_data.cell_state, quantized_cell_state,
      batch_size * state_dimension,
      quantization_settings.cell_quantization_parameters.scale,
      quantization_settings.cell_quantization_parameters.zero_point);

  CellType quantized_forget_gate[batch_size * state_dimension] = {};
  tflite::Quantize(gate_output_data.expected_forget_gate_output,
                   quantized_forget_gate, batch_size * state_dimension,
                   quantization_settings.nonlinear_activation_output_scale, 0);

  CellType quantized_input_gate[batch_size * state_dimension] = {};
  tflite::Quantize(gate_output_data.expected_input_gate_output,
                   quantized_input_gate, batch_size * state_dimension,
                   quantization_settings.nonlinear_activation_output_scale, 0);

  CellType quantized_cell_gate[batch_size * state_dimension] = {};
  tflite::Quantize(gate_output_data.expected_cell_gate_output,
                   quantized_cell_gate, batch_size * state_dimension,
                   quantization_settings.nonlinear_activation_output_scale, 0);

  tflite::lstm_internal::UpdateLstmCellInteger(
      batch_size, state_dimension, quantized_cell_state, cell_scale_shift,
      quantized_input_gate, quantized_forget_gate, quantized_cell_gate, false,
      quantized_cell_clip);

  float cell_state_float[batch_size * state_dimension] = {};
  Dequantize(quantized_cell_state, batch_size * state_dimension,
             quantization_settings.cell_quantization_parameters.scale,
             quantization_settings.cell_quantization_parameters.zero_point,
             cell_state_float);

  ValidateResultGoldens(gate_output_data.expected_updated_cell,
                        cell_state_float, batch_size * state_dimension,
                        tolerance);
}

template <int batch_size, int input_dimension, int state_dimension>
void TestHiddenStateUpdateFloat(
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    const float tolerance) {
  // If no projection layer, hidden state dimension == output dimension ==
  // cell state dimension
  float output[batch_size * state_dimension] = {};
  float scratch[batch_size * state_dimension] = {};

  tflite::lstm_internal::CalculateLstmOutputFloat(
      batch_size, state_dimension, state_dimension,
      gate_output_data.expected_updated_cell,
      gate_output_data.expected_output_gate_output, kTfLiteActTanh, nullptr,
      nullptr, 0, output, scratch);

  ValidateResultGoldens(gate_output_data.expected_updated_hidden, output,
                        batch_size * state_dimension, tolerance);
}

template <typename ActivationType, typename BiasType, typename CellType,
          int batch_size, int input_dimension, int state_dimension>
void TestHiddenStateUpdateQuantized(
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    const ModelQuantizationParameters& quantization_settings,
    const IntegerLstmParameter& evaluation_params, const float tolerance) {
  CellType quantized_cell_state[batch_size * state_dimension] = {};
  tflite::Quantize(
      gate_output_data.expected_updated_cell, quantized_cell_state,
      batch_size * state_dimension,
      quantization_settings.cell_quantization_parameters.scale,
      quantization_settings.cell_quantization_parameters.zero_point);

  CellType quantized_output_gate[batch_size * state_dimension] = {};
  tflite::Quantize(gate_output_data.expected_output_gate_output,
                   quantized_output_gate, batch_size * state_dimension,
                   quantization_settings.nonlinear_activation_output_scale, 0);

  // scratches
  int16_t scratch0[batch_size * state_dimension] = {};
  int8_t scratch1[batch_size * state_dimension] = {};
  int32_t scratch2[batch_size * state_dimension] = {};

  // output (updated hidden state)
  int8_t output_state[batch_size * state_dimension] = {};

  tflite::lstm_internal::CalculateLstmOutputInteger8x8_16(
      batch_size, state_dimension, state_dimension, quantized_cell_state,
      evaluation_params.cell_scale, quantized_output_gate,
      evaluation_params.effective_hidden_scale_a,
      evaluation_params.effective_hidden_scale_b, evaluation_params.hidden_zp,
      /*projection_weights=*/nullptr, /*proj_scale_a=*/0, 0, 0,
      /*output_state_zp=*/evaluation_params.hidden_zp,
      evaluation_params.quantized_proj_clip, output_state, scratch0, scratch1,
      scratch2);

  float output_state_float[batch_size * state_dimension] = {};
  Dequantize(output_state, batch_size * state_dimension,
             quantization_settings.hidden_quantization_parameters.scale,
             quantization_settings.hidden_quantization_parameters.zero_point,
             output_state_float);

  ValidateResultGoldens(gate_output_data.expected_updated_hidden,
                        output_state_float, batch_size * state_dimension,
                        tolerance);
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestOneStepLSTMFloat(
    const TfLiteLSTMParams& general_model_settings,
    ModelContents<float, float, float, float, batch_size, time_steps,
                  input_dimension, state_dimension>& model_contents,
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    const float tolerance) {
  // scratch buffers
  float forget_gate_scratch[batch_size * state_dimension] = {};
  float input_gate_scratch[batch_size * state_dimension] = {};
  float cell_gate_scratch[batch_size * state_dimension] = {};
  float output_gate_scratch[batch_size * state_dimension] = {};

  tflite::lstm_internal::LstmStepFloat(
      gate_output_data.input_data,
      model_contents.InputGateParams().activation_weight,
      model_contents.ForgetGateParams().activation_weight,
      model_contents.CellGateParams().activation_weight,
      model_contents.OutputGateParams().activation_weight,
      /*aux_input_ptr=*/nullptr, /*aux_input_to_input_weights_ptr=*/nullptr,
      /*aux_input_to_forget_weights_ptr=*/nullptr,
      /*aux_input_to_cell_weights_ptr=*/nullptr,
      /*aux_input_to_output_weights_ptr=*/nullptr,
      model_contents.InputGateParams().recurrent_weight,
      model_contents.ForgetGateParams().recurrent_weight,
      model_contents.CellGateParams().recurrent_weight,
      model_contents.OutputGateParams().recurrent_weight,
      /*cell_to_input_weights_ptr=*/nullptr,
      /*cell_to_forget_weights_ptr=*/nullptr,
      /*cell_to_output_weights_ptr=*/nullptr,
      /*input_layer_norm_coefficients_ptr=*/nullptr,
      /*forget_layer_norm_coefficients_ptr=*/nullptr,
      /*cell_layer_norm_coefficients_ptr=*/nullptr,
      /*output_layer_norm_coefficients_ptr=*/nullptr,
      model_contents.InputGateParams().fused_bias,
      model_contents.ForgetGateParams().fused_bias,
      model_contents.CellGateParams().fused_bias,
      model_contents.OutputGateParams().fused_bias,
      /*projection_weights_ptr=*/nullptr, /*projection_bias_ptr=*/nullptr,
      &general_model_settings, batch_size, state_dimension, input_dimension,
      input_dimension, state_dimension,
      /*output_batch_leading_dim=*/0, model_contents.GetHiddenState(),
      model_contents.GetCellState(), input_gate_scratch, forget_gate_scratch,
      cell_gate_scratch, output_gate_scratch, model_contents.GetOutput());

  ValidateResultGoldens(gate_output_data.expected_updated_hidden,
                        model_contents.GetHiddenState(),
                        batch_size * state_dimension, tolerance);
  ValidateResultGoldens(gate_output_data.expected_updated_cell,
                        model_contents.GetCellState(),
                        batch_size * state_dimension, tolerance);
}

template <typename ActivationType, typename BiasType, typename CellType,
          int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestOneStepLSTMQuantized(
    ModelContents<ActivationType, int8_t, BiasType, CellType, batch_size,
                  time_steps, input_dimension, state_dimension>& model_contents,
    const ModelQuantizationParameters& quantization_settings,
    const IntegerLstmParameter& evaluation_params,
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    const float hidden_state_tolerance, const float cell_state_tolerance) {
  // Scratch buffers
  CellType scratch0[batch_size * state_dimension] = {};
  CellType scratch1[batch_size * state_dimension] = {};
  CellType scratch2[batch_size * state_dimension] = {};
  CellType scratch3[batch_size * state_dimension] = {};
  ActivationType scratch4[batch_size * state_dimension] = {};
  BiasType scratch5[batch_size * state_dimension] = {};

  tflite::lstm_internal::LstmStepInteger8x8_16(
      model_contents.GetInput(),
      model_contents.InputGateParams().activation_weight,
      evaluation_params.effective_input_to_input_scale_a,
      evaluation_params.effective_input_to_input_scale_b,
      model_contents.ForgetGateParams().activation_weight,
      evaluation_params.effective_input_to_forget_scale_a,
      evaluation_params.effective_input_to_forget_scale_b,
      model_contents.CellGateParams().activation_weight,
      evaluation_params.effective_input_to_cell_scale_a,
      evaluation_params.effective_input_to_cell_scale_b,
      model_contents.OutputGateParams().activation_weight,
      evaluation_params.effective_input_to_output_scale_a,
      evaluation_params.effective_input_to_output_scale_b,
      model_contents.InputGateParams().recurrent_weight,
      evaluation_params.effective_recurrent_to_input_scale_a,
      evaluation_params.effective_recurrent_to_input_scale_b,
      model_contents.ForgetGateParams().recurrent_weight,
      evaluation_params.effective_recurrent_to_forget_scale_a,
      evaluation_params.effective_recurrent_to_forget_scale_b,
      model_contents.CellGateParams().recurrent_weight,
      evaluation_params.effective_recurrent_to_cell_scale_a,
      evaluation_params.effective_recurrent_to_cell_scale_b,
      model_contents.OutputGateParams().recurrent_weight,
      evaluation_params.effective_recurrent_to_output_scale_a,
      evaluation_params.effective_recurrent_to_output_scale_b,
      /*cell_to_input_weight_ptr=*/nullptr,
      /*effective_cell_to_input_scale_a=*/0,
      /*effective_cell_to_input_scale_b=*/0,
      /*cell_to_forget_weight_ptr=*/nullptr,
      /*effective_cell_to_forget_scale_a=*/0,
      /*effective_cell_to_forget_scale_b=*/0,
      /*cell_to_output_weight_ptr=*/nullptr,
      /*effective_cell_to_output_scale_a=*/0,
      /*effective_cell_to_output_scale_b=*/0,
      /*projection_weight_ptr=*/nullptr, /*effective_proj_scale_a=*/0,
      /*effective_proj_scale_b=*/0, evaluation_params.hidden_zp,
      evaluation_params.effective_hidden_scale_a,
      evaluation_params.effective_hidden_scale_b,
      /*layer_norm_input_weight_ptr=*/nullptr,
      /*layer_norm_input_scale_a=*/0, /*layer_norm_input_scale_b=*/0,
      /*layer_norm_forget_weight_ptr=*/nullptr,
      /*layer_norm_forget_scale_a=*/0, /*layer_norm_forget_scale_b=*/0,
      /*layer_norm_cell_weight_ptr=*/nullptr,
      /*layer_norm_cell_scale_a=*/0, /*layer_norm_cell_scale_b=*/0,
      /*layer_norm_output_weight_ptr=*/nullptr,
      /*layer_norm_output_scale_a=*/0, /*layer_norm_output_scale_b=*/0,
      /*input_gate_bias_ptr=*/nullptr, /*forget_gate_bias_ptr=*/nullptr,
      /*cell_gate_bias_ptr=*/nullptr, /*output_gate_bias_ptr=*/nullptr,
      evaluation_params.quantized_cell_clip,
      evaluation_params.quantized_proj_clip, evaluation_params.cell_scale,
      /*input_variance_guard=*/0, /*forget_variance_guard=*/0,
      /*cell_variance_guard=*/0, /*output_variance_guard=*/0,
      evaluation_params.input_to_forget_effective_bias,
      evaluation_params.recurrent_to_forget_effective_bias,
      evaluation_params.input_to_cell_effective_bias,
      evaluation_params.recurrent_to_cell_effective_bias,
      evaluation_params.input_to_output_effective_bias,
      evaluation_params.recurrent_to_output_effective_bias,
      evaluation_params.input_to_input_effective_bias,
      evaluation_params.recurrent_to_input_effective_bias,
      evaluation_params.projection_effective_bias, batch_size, input_dimension,
      state_dimension, state_dimension, model_contents.GetHiddenState(),
      quantization_settings.output_quantization_parameters.zero_point,
      model_contents.GetCellState(), model_contents.GetOutput(), scratch0,
      scratch1, scratch2, scratch3, scratch4, scratch5);

  float dequantized_hidden_state[batch_size * state_dimension] = {};
  Dequantize(model_contents.GetHiddenState(), batch_size * state_dimension,
             quantization_settings.hidden_quantization_parameters.scale,
             quantization_settings.hidden_quantization_parameters.zero_point,
             dequantized_hidden_state);

  float dequantized_cell_state[batch_size * state_dimension] = {};
  Dequantize(model_contents.GetCellState(), batch_size * state_dimension,
             quantization_settings.cell_quantization_parameters.scale,
             quantization_settings.cell_quantization_parameters.zero_point,
             dequantized_cell_state);

  ValidateResultGoldens(gate_output_data.expected_updated_hidden,
                        dequantized_hidden_state, batch_size * state_dimension,
                        hidden_state_tolerance);
  ValidateResultGoldens(gate_output_data.expected_updated_cell,
                        dequantized_cell_state, batch_size * state_dimension,
                        cell_state_tolerance);
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestLSTMEvalFloat(
    const TfLiteLSTMParams& general_model_settings,
    ModelContents<float, float, float, float, batch_size, time_steps,
                  input_dimension, state_dimension>& float_model_contents,
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float tolerance) {
  float scratch_buffers[4 * batch_size * state_dimension] = {};

  tflite::EvalFloatLstm(
      float_model_contents.GetInternalTensor(kLstmInputTensor),
      float_model_contents.GetInternalTensor(kLstmInputToInputWeightsTensor),
      float_model_contents.GetInternalTensor(kLstmInputToForgetWeightsTensor),
      float_model_contents.GetInternalTensor(kLstmInputToCellWeightsTensor),
      float_model_contents.GetInternalTensor(kLstmInputToOutputWeightsTensor),
      float_model_contents.GetInternalTensor(
          kLstmRecurrentToInputWeightsTensor),
      float_model_contents.GetInternalTensor(
          kLstmRecurrentToForgetWeightsTensor),
      float_model_contents.GetInternalTensor(kLstmRecurrentToCellWeightsTensor),
      float_model_contents.GetInternalTensor(
          kLstmRecurrentToOutputWeightsTensor),
      /*cell_to_input_weights=*/nullptr,
      /*cell_to_forget_weights=*/nullptr,
      /*cell_to_output_weights=*/nullptr,
      /*input_layer_norm_coefficients=*/nullptr,
      /*forget_layer_norm_coefficients=*/nullptr,
      /*cell_layer_norm_coefficients=*/nullptr,
      /*output_layer_norm_coefficients=*/nullptr,
      /*aux_input=*/nullptr,
      /*aux_input_to_input_weights=*/nullptr,
      /*aux_input_to_forget_weights=*/nullptr,
      /*aux_input_to_cell_weights=*/nullptr,
      /*aux_input_to_output_weights=*/nullptr,
      float_model_contents.GetInternalTensor(kLstmInputGateBiasTensor),
      float_model_contents.GetInternalTensor(kLstmForgetGateBiasTensor),
      float_model_contents.GetInternalTensor(kLstmCellGateBiasTensor),
      float_model_contents.GetInternalTensor(kLstmOutputGateBiasTensor),
      /*projection_weights=*/nullptr,
      /*projection_bias=*/nullptr, &general_model_settings,
      /*forward_sequence=*/true, /*time_major=*/false,
      /*output_offset=*/0, scratch_buffers,
      float_model_contents.HiddenStateTensor(),
      float_model_contents.CellStateTensor(),
      float_model_contents.OutputTensor());

  // Validate hidden state. See previous test for the calculation
  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        float_model_contents.GetHiddenState(),
                        batch_size * state_dimension, tolerance);
  // Validate cell state. See previous test for the calculation
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        float_model_contents.GetCellState(),
                        batch_size * state_dimension, tolerance);
  // Validate output . See previous test for the calculation
  ValidateResultGoldens(eval_check_data.expected_output,
                        float_model_contents.GetOutput(),
                        batch_size * state_dimension * time_steps, tolerance);
}

template <typename ActivationType, typename BiasType, typename CellType,
          int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestLSTMEvalQuantized(
    const TfLiteLSTMParams& general_model_settings,
    ModelContents<ActivationType, int8_t, BiasType, CellType, batch_size,
                  time_steps, input_dimension, state_dimension>&
        quantized_model_content,
    const ModelQuantizationParameters& quantization_settings,
    const IntegerLstmParameter& evaluation_params,
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float hidden_state_tolerance, const float cell_state_tolerance) {
  // Scratch buffers
  CellType scratch0[batch_size * state_dimension] = {};
  CellType scratch1[batch_size * state_dimension] = {};
  CellType scratch2[batch_size * state_dimension] = {};
  CellType scratch3[batch_size * state_dimension] = {};
  ActivationType scratch4[batch_size * state_dimension * time_steps] = {};
  BiasType scratch5[batch_size * state_dimension] = {};

  EvalInteger8x8_16Lstm(
      quantized_model_content.GetInternalTensor(kLstmInputTensor),
      quantized_model_content.GetInternalTensor(kLstmInputToInputWeightsTensor),
      quantized_model_content.GetInternalTensor(
          kLstmInputToForgetWeightsTensor),
      quantized_model_content.GetInternalTensor(kLstmInputToCellWeightsTensor),
      quantized_model_content.GetInternalTensor(
          kLstmInputToOutputWeightsTensor),
      quantized_model_content.GetInternalTensor(
          kLstmRecurrentToInputWeightsTensor),
      quantized_model_content.GetInternalTensor(
          kLstmRecurrentToForgetWeightsTensor),
      quantized_model_content.GetInternalTensor(
          kLstmRecurrentToCellWeightsTensor),
      quantized_model_content.GetInternalTensor(
          kLstmRecurrentToOutputWeightsTensor),
      /*cell_to_input_weights=*/nullptr,
      /*cell_to_forget_weights=*/nullptr,
      /*cell_to_output_weights=*/nullptr,
      /*input_layer_norm_coefficients=*/nullptr,
      /*forget_layer_norm_coefficients=*/nullptr,
      /*cell_layer_norm_coefficients=*/nullptr,
      /*output_layer_norm_coefficients=*/nullptr,
      quantized_model_content.GetInternalTensor(kLstmInputGateBiasTensor),
      quantized_model_content.GetInternalTensor(kLstmForgetGateBiasTensor),
      quantized_model_content.GetInternalTensor(kLstmCellGateBiasTensor),
      quantized_model_content.GetInternalTensor(kLstmOutputGateBiasTensor),
      /*projection_weights=*/nullptr,
      /*projection_bias=*/nullptr, &general_model_settings,
      /*forward_sequence=*/true, /*time_major=*/false, &evaluation_params,
      quantization_settings.output_quantization_parameters.zero_point,
      quantized_model_content.HiddenStateTensor(),
      quantized_model_content.CellStateTensor(),
      quantized_model_content.OutputTensor(), scratch0, scratch1, scratch2,
      scratch3, scratch4, scratch5);

  float dequantized_hidden_state[batch_size * state_dimension] = {};
  Dequantize(quantized_model_content.GetHiddenState(),
             batch_size * state_dimension,
             quantization_settings.hidden_quantization_parameters.scale,
             quantization_settings.hidden_quantization_parameters.zero_point,
             dequantized_hidden_state);

  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        dequantized_hidden_state, batch_size * state_dimension,
                        hidden_state_tolerance);

  float dequantized_cell_state[batch_size * state_dimension] = {};
  Dequantize(quantized_model_content.GetCellState(),
             batch_size * state_dimension,
             quantization_settings.cell_quantization_parameters.scale,
             quantization_settings.cell_quantization_parameters.zero_point,
             dequantized_cell_state);
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        dequantized_cell_state, batch_size * state_dimension,
                        cell_state_tolerance);

  float dequantized_output[batch_size * state_dimension * time_steps] = {};
  Dequantize(quantized_model_content.GetOutput(),
             batch_size * state_dimension * time_steps,
             quantization_settings.output_quantization_parameters.scale,
             quantization_settings.output_quantization_parameters.zero_point,
             dequantized_output);
  ValidateResultGoldens(eval_check_data.expected_output, dequantized_output,
                        batch_size * state_dimension, hidden_state_tolerance);
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_TEST_COMMOM_H_
