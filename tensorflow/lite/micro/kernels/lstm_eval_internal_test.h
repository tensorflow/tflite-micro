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
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval_internal.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

// IntegerLstmParameter is required by the legend int8 code. Not required for
// the generalized standard LSTM (e.g., 16bits activation case)
template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
IntegerLstmParameter CreateIntegerParameter(
    const LstmNodeContent<int8_t, int8_t, int32_t, int16_t, batch_size,
                          time_steps, input_dimension, state_dimension>&
        quantized_node_contents) {
  IntegerLstmParameter evaluation_params;
  double effective_scale;
  int buffer_shift_output;

  const auto quantization_settings =
      quantized_node_contents.QuantizationSettings();
  effective_scale = quantization_settings.input.scale *
                    quantization_settings.forget_gate.activation_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_input_to_forget_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_input_to_forget_scale_b = buffer_shift_output;
  effective_scale = quantization_settings.output.scale *
                    quantization_settings.forget_gate.recurrent_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_recurrent_to_forget_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_recurrent_to_forget_scale_b = buffer_shift_output;
  // Set effective bias
  evaluation_params.input_to_forget_effective_bias = const_cast<int32_t*>(
      quantized_node_contents.ForgetGateData().activation_zp_folded_bias);
  evaluation_params.recurrent_to_forget_effective_bias = const_cast<int32_t*>(
      quantized_node_contents.ForgetGateData().recurrent_zp_folded_bias);

  // input gate
  effective_scale = quantization_settings.input.scale *
                    quantization_settings.input_gate.activation_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_input_to_input_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_input_to_input_scale_b = buffer_shift_output;
  effective_scale = quantization_settings.output.scale *
                    quantization_settings.input_gate.recurrent_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_recurrent_to_input_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_recurrent_to_input_scale_b = buffer_shift_output;
  // Set effective bias
  evaluation_params.input_to_input_effective_bias = const_cast<int32_t*>(
      quantized_node_contents.InputGateData().activation_zp_folded_bias);
  evaluation_params.recurrent_to_input_effective_bias = const_cast<int32_t*>(
      quantized_node_contents.InputGateData().recurrent_zp_folded_bias);

  // cell gate
  effective_scale = quantization_settings.input.scale *
                    quantization_settings.cell_gate.activation_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_input_to_cell_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_input_to_cell_scale_b = buffer_shift_output;
  effective_scale = quantization_settings.output.scale *
                    quantization_settings.cell_gate.recurrent_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_recurrent_to_cell_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_recurrent_to_cell_scale_b = buffer_shift_output;
  // Set effective bias
  evaluation_params.input_to_cell_effective_bias = const_cast<int32_t*>(
      quantized_node_contents.CellGateData().activation_zp_folded_bias);
  evaluation_params.recurrent_to_cell_effective_bias = const_cast<int32_t*>(
      quantized_node_contents.CellGateData().recurrent_zp_folded_bias);

  // output gate
  effective_scale = quantization_settings.input.scale *
                    quantization_settings.output_gate.activation_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_input_to_output_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_input_to_output_scale_b = buffer_shift_output;
  effective_scale = quantization_settings.output.scale *
                    quantization_settings.output_gate.recurrent_weight.scale /
                    quantization_settings.nonlinear_activation_input_scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_recurrent_to_output_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_recurrent_to_output_scale_b = buffer_shift_output;
  // Set effective bias
  evaluation_params.input_to_output_effective_bias = const_cast<int32_t*>(
      quantized_node_contents.OutputGateData().activation_zp_folded_bias);
  evaluation_params.recurrent_to_output_effective_bias = const_cast<int32_t*>(
      quantized_node_contents.OutputGateData().recurrent_zp_folded_bias);

  // hidden state (no projection, output is the hidden state)
  effective_scale = quantization_settings.nonlinear_activation_output_scale *
                    quantization_settings.nonlinear_activation_output_scale /
                    quantization_settings.hidden_state.scale;
  QuantizeMultiplier(effective_scale,
                     &evaluation_params.effective_hidden_scale_a,
                     &buffer_shift_output);
  evaluation_params.effective_hidden_scale_b = buffer_shift_output;
  evaluation_params.hidden_zp = quantization_settings.hidden_state.zero_point;

  // cell state. Note, cell_scale is actually not a scale. 2^-cell_scale is
  // the true scale for cell
  int buffer_cell_scale;
  tflite::CheckedLog2(quantization_settings.cell_state.scale,
                      &buffer_cell_scale);
  evaluation_params.cell_scale = buffer_cell_scale;

  evaluation_params.quantized_cell_clip = static_cast<int16_t>(std::min(
      std::max(
          static_cast<double>(quantized_node_contents.BuiltinData().cell_clip) /
              quantization_settings.cell_state.scale,
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
void TestGateOutputFloat(
    const GateData<float, float, input_dimension, state_dimension>& gate_params,
    const TfLiteFusedActivation activation_type, const float* input_data,
    const float* hidden_state, const float* expected_vals,
    const float tolerance) {
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

template <typename ActivationType, typename BiasType, typename CellType,
          int batch_size, int input_dimension, int state_dimension>
void TestGateOutputQuantized(
    const ActivationType* quantized_input,
    const ActivationType* quantized_hidden_state,
    const GateData<int8_t, BiasType, input_dimension, state_dimension>&
        gate_params,
    const NodeQuantizationParameters& quantization_settings,
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
    const NodeQuantizationParameters& quantization_settings,
    const int32_t cell_scale_shift, const CellType quantized_cell_clip,
    const float tolerance) {
  CellType quantized_cell_state[batch_size * state_dimension] = {};
  tflite::Quantize(gate_output_data.cell_state, quantized_cell_state,
                   batch_size * state_dimension,
                   quantization_settings.cell_state.scale,
                   quantization_settings.cell_state.zero_point);

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
             quantization_settings.cell_state.scale,
             quantization_settings.cell_state.zero_point, cell_state_float);

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
    const NodeQuantizationParameters& quantization_settings,
    const IntegerLstmParameter& evaluation_params, const float tolerance) {
  CellType quantized_cell_state[batch_size * state_dimension] = {};
  tflite::Quantize(gate_output_data.expected_updated_cell, quantized_cell_state,
                   batch_size * state_dimension,
                   quantization_settings.cell_state.scale,
                   quantization_settings.cell_state.zero_point);

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
             quantization_settings.hidden_state.scale,
             quantization_settings.hidden_state.zero_point, output_state_float);

  ValidateResultGoldens(gate_output_data.expected_updated_hidden,
                        output_state_float, batch_size * state_dimension,
                        tolerance);
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestOneStepLSTMFloat(
    /*can not be const, state will be updated*/
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& node_contents,
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    const float tolerance) {
  // scratch buffers
  float forget_gate_scratch[batch_size * state_dimension] = {};
  float input_gate_scratch[batch_size * state_dimension] = {};
  float cell_gate_scratch[batch_size * state_dimension] = {};
  float output_gate_scratch[batch_size * state_dimension] = {};

  // states and output will be modified (cannot use the const getter)
  float* hidden_state = node_contents.GetHiddenStateData();
  float* cell_state = node_contents.GetCellStateData();
  float* output = node_contents.GetOutputData();

  const auto builtin_data = node_contents.BuiltinData();
  // Copy out the LSTM specific params so they can be passed in the function.
  TfLiteLSTMParams general_model_settings;
  general_model_settings.activation = builtin_data.activation;
  general_model_settings.cell_clip = builtin_data.cell_clip;
  general_model_settings.proj_clip = builtin_data.proj_clip;
  general_model_settings.asymmetric_quantize_inputs =
      builtin_data.asymmetric_quantize_inputs;

  tflite::lstm_internal::LstmStepFloat(
      gate_output_data.input_data,
      node_contents.InputGateData().activation_weight,
      node_contents.ForgetGateData().activation_weight,
      node_contents.CellGateData().activation_weight,
      node_contents.OutputGateData().activation_weight,
      /*aux_input_ptr=*/nullptr, /*aux_input_to_input_weights_ptr=*/nullptr,
      /*aux_input_to_forget_weights_ptr=*/nullptr,
      /*aux_input_to_cell_weights_ptr=*/nullptr,
      /*aux_input_to_output_weights_ptr=*/nullptr,
      node_contents.InputGateData().recurrent_weight,
      node_contents.ForgetGateData().recurrent_weight,
      node_contents.CellGateData().recurrent_weight,
      node_contents.OutputGateData().recurrent_weight,
      /*cell_to_input_weights_ptr=*/nullptr,
      /*cell_to_forget_weights_ptr=*/nullptr,
      /*cell_to_output_weights_ptr=*/nullptr,
      /*input_layer_norm_coefficients_ptr=*/nullptr,
      /*forget_layer_norm_coefficients_ptr=*/nullptr,
      /*cell_layer_norm_coefficients_ptr=*/nullptr,
      /*output_layer_norm_coefficients_ptr=*/nullptr,
      node_contents.InputGateData().fused_bias,
      node_contents.ForgetGateData().fused_bias,
      node_contents.CellGateData().fused_bias,
      node_contents.OutputGateData().fused_bias,
      /*projection_weights_ptr=*/nullptr, /*projection_bias_ptr=*/nullptr,
      &general_model_settings, batch_size, state_dimension, input_dimension,
      input_dimension, state_dimension,
      /*output_batch_leading_dim=*/0, hidden_state, cell_state,
      input_gate_scratch, forget_gate_scratch, cell_gate_scratch,
      output_gate_scratch, output);

  ValidateResultGoldens(gate_output_data.expected_updated_hidden, hidden_state,
                        batch_size * state_dimension, tolerance);
  ValidateResultGoldens(gate_output_data.expected_updated_cell, cell_state,
                        batch_size * state_dimension, tolerance);
}

template <typename ActivationType, typename BiasType, typename CellType,
          int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestOneStepLSTMQuantized(
    /*can not be const, state will be updated*/
    LstmNodeContent<ActivationType, int8_t, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        model_contents,
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

  // states and output will be modified (cannot use the const getter)
  ActivationType* hidden_state = model_contents.GetHiddenStateData();
  CellType* cell_state = model_contents.GetCellStateData();
  ActivationType* output = model_contents.GetOutputData();

  const auto evaluation_params =
      tflite::testing::CreateIntegerParameter(model_contents);
  const auto quantization_settings = model_contents.QuantizationSettings();

  tflite::lstm_internal::LstmStepInteger8x8_16(
      model_contents.GetInputData(),
      model_contents.InputGateData().activation_weight,
      evaluation_params.effective_input_to_input_scale_a,
      evaluation_params.effective_input_to_input_scale_b,
      model_contents.ForgetGateData().activation_weight,
      evaluation_params.effective_input_to_forget_scale_a,
      evaluation_params.effective_input_to_forget_scale_b,
      model_contents.CellGateData().activation_weight,
      evaluation_params.effective_input_to_cell_scale_a,
      evaluation_params.effective_input_to_cell_scale_b,
      model_contents.OutputGateData().activation_weight,
      evaluation_params.effective_input_to_output_scale_a,
      evaluation_params.effective_input_to_output_scale_b,
      model_contents.InputGateData().recurrent_weight,
      evaluation_params.effective_recurrent_to_input_scale_a,
      evaluation_params.effective_recurrent_to_input_scale_b,
      model_contents.ForgetGateData().recurrent_weight,
      evaluation_params.effective_recurrent_to_forget_scale_a,
      evaluation_params.effective_recurrent_to_forget_scale_b,
      model_contents.CellGateData().recurrent_weight,
      evaluation_params.effective_recurrent_to_cell_scale_a,
      evaluation_params.effective_recurrent_to_cell_scale_b,
      model_contents.OutputGateData().recurrent_weight,
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
      state_dimension, state_dimension, hidden_state,
      quantization_settings.output.zero_point, cell_state, output, scratch0,
      scratch1, scratch2, scratch3, scratch4, scratch5);

  float dequantized_hidden_state[batch_size * state_dimension] = {};
  Dequantize(hidden_state, batch_size * state_dimension,
             quantization_settings.hidden_state.scale,
             quantization_settings.hidden_state.zero_point,
             dequantized_hidden_state);

  float dequantized_cell_state[batch_size * state_dimension] = {};
  Dequantize(cell_state, batch_size * state_dimension,
             quantization_settings.cell_state.scale,
             quantization_settings.cell_state.zero_point,
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
    /*can not be const, state will be updated*/
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& float_model_contents,
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float tolerance) {
  float scratch_buffers[4 * batch_size * state_dimension] = {};

  const auto builtin_data = float_model_contents.BuiltinData();
  // Copy out the LSTM specific params so they can be passed in the function.
  TfLiteLSTMParams general_model_settings;
  general_model_settings.activation = builtin_data.activation;
  general_model_settings.cell_clip = builtin_data.cell_clip;
  general_model_settings.proj_clip = builtin_data.proj_clip;
  general_model_settings.asymmetric_quantize_inputs =
      builtin_data.asymmetric_quantize_inputs;

  tflite::EvalFloatLstm(
      float_model_contents.GetEvalTensor(kLstmInputTensor),
      float_model_contents.GetEvalTensor(kLstmInputToInputWeightsTensor),
      float_model_contents.GetEvalTensor(kLstmInputToForgetWeightsTensor),
      float_model_contents.GetEvalTensor(kLstmInputToCellWeightsTensor),
      float_model_contents.GetEvalTensor(kLstmInputToOutputWeightsTensor),
      float_model_contents.GetEvalTensor(kLstmRecurrentToInputWeightsTensor),
      float_model_contents.GetEvalTensor(kLstmRecurrentToForgetWeightsTensor),
      float_model_contents.GetEvalTensor(kLstmRecurrentToCellWeightsTensor),
      float_model_contents.GetEvalTensor(kLstmRecurrentToOutputWeightsTensor),
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
      float_model_contents.GetEvalTensor(kLstmInputGateBiasTensor),
      float_model_contents.GetEvalTensor(kLstmForgetGateBiasTensor),
      float_model_contents.GetEvalTensor(kLstmCellGateBiasTensor),
      float_model_contents.GetEvalTensor(kLstmOutputGateBiasTensor),
      /*projection_weights=*/nullptr,
      /*projection_bias=*/nullptr, &general_model_settings,
      /*forward_sequence=*/true, /*time_major=*/false,
      /*output_offset=*/0, scratch_buffers,
      float_model_contents.HiddenStateEvalTensor(),
      float_model_contents.CellStateEvalTensor(),
      float_model_contents.OutputEvalTensor());

  // Validate hidden state. See previous test for the calculation
  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        float_model_contents.GetHiddenStateData(),
                        batch_size * state_dimension, tolerance);
  // Validate cell state. See previous test for the calculation
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        float_model_contents.GetCellStateData(),
                        batch_size * state_dimension, tolerance);
  // Validate output . See previous test for the calculation
  ValidateResultGoldens(eval_check_data.expected_output,
                        float_model_contents.GetOutputData(),
                        batch_size * state_dimension * time_steps, tolerance);
}

template <typename ActivationType, typename BiasType, typename CellType,
          int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestLSTMEvalQuantized(
    /*can not be const, state will be updated*/
    LstmNodeContent<ActivationType, int8_t, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        quantized_model_content,
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

  const auto quantization_settings =
      quantized_model_content.QuantizationSettings();
  const auto evaluation_params =
      tflite::testing::CreateIntegerParameter(quantized_model_content);
  const auto builtin_data = quantized_model_content.BuiltinData();

  // Copy out the LSTM specific params so they can be passed in the function.
  TfLiteLSTMParams general_model_settings;
  general_model_settings.activation = builtin_data.activation;
  general_model_settings.cell_clip = builtin_data.cell_clip;
  general_model_settings.proj_clip = builtin_data.proj_clip;
  general_model_settings.asymmetric_quantize_inputs =
      builtin_data.asymmetric_quantize_inputs;

  EvalInteger8x8_16Lstm(
      quantized_model_content.GetEvalTensor(kLstmInputTensor),
      quantized_model_content.GetEvalTensor(kLstmInputToInputWeightsTensor),
      quantized_model_content.GetEvalTensor(kLstmInputToForgetWeightsTensor),
      quantized_model_content.GetEvalTensor(kLstmInputToCellWeightsTensor),
      quantized_model_content.GetEvalTensor(kLstmInputToOutputWeightsTensor),
      quantized_model_content.GetEvalTensor(kLstmRecurrentToInputWeightsTensor),
      quantized_model_content.GetEvalTensor(
          kLstmRecurrentToForgetWeightsTensor),
      quantized_model_content.GetEvalTensor(kLstmRecurrentToCellWeightsTensor),
      quantized_model_content.GetEvalTensor(
          kLstmRecurrentToOutputWeightsTensor),
      /*cell_to_input_weights=*/nullptr,
      /*cell_to_forget_weights=*/nullptr,
      /*cell_to_output_weights=*/nullptr,
      /*input_layer_norm_coefficients=*/nullptr,
      /*forget_layer_norm_coefficients=*/nullptr,
      /*cell_layer_norm_coefficients=*/nullptr,
      /*output_layer_norm_coefficients=*/nullptr,
      quantized_model_content.GetEvalTensor(kLstmInputGateBiasTensor),
      quantized_model_content.GetEvalTensor(kLstmForgetGateBiasTensor),
      quantized_model_content.GetEvalTensor(kLstmCellGateBiasTensor),
      quantized_model_content.GetEvalTensor(kLstmOutputGateBiasTensor),
      /*projection_weights=*/nullptr,
      /*projection_bias=*/nullptr, &general_model_settings,
      /*forward_sequence=*/true, /*time_major=*/false, &evaluation_params,
      quantization_settings.output.zero_point,
      quantized_model_content.HiddenStateEvalTensor(),
      quantized_model_content.CellStateEvalTensor(),
      quantized_model_content.OutputEvalTensor(), scratch0, scratch1, scratch2,
      scratch3, scratch4, scratch5);

  float dequantized_hidden_state[batch_size * state_dimension] = {};
  Dequantize(
      quantized_model_content.GetHiddenStateData(),
      batch_size * state_dimension, quantization_settings.hidden_state.scale,
      quantization_settings.hidden_state.zero_point, dequantized_hidden_state);

  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        dequantized_hidden_state, batch_size * state_dimension,
                        hidden_state_tolerance);

  float dequantized_cell_state[batch_size * state_dimension] = {};
  Dequantize(
      quantized_model_content.GetCellStateData(), batch_size * state_dimension,
      quantization_settings.cell_state.scale,
      quantization_settings.cell_state.zero_point, dequantized_cell_state);
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        dequantized_cell_state, batch_size * state_dimension,
                        cell_state_tolerance);

  float dequantized_output[batch_size * state_dimension * time_steps] = {};
  Dequantize(quantized_model_content.GetOutputData(),
             batch_size * state_dimension * time_steps,
             quantization_settings.output.scale,
             quantization_settings.output.zero_point, dequantized_output);
  ValidateResultGoldens(eval_check_data.expected_output, dequantized_output,
                        batch_size * state_dimension, hidden_state_tolerance);
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_TEST_COMMOM_H_
