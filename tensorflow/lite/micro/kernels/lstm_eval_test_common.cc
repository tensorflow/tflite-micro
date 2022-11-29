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

#include "tensorflow/lite/micro/kernels/lstm_eval_test.h"

namespace tflite {
namespace testing {
TestModelContents<float, float, float, float, 2, 3, 2, 2>
Create2x3x2X2FloatModelContents() {
  // Parameters for different gates
  // negative large weights for forget gate to make it really forget
  const GateParameters<float, float, 2, 2> forget_gate_params = {
      /*.activation_weight=*/{-10, -10, -20, -20},
      /*.recurrent_weight=*/{-10, -10, -20, -20},
      /*.fused_bias=*/{1, 2},
      /*activation_zp_folded_bias=*/{0, 0},
      /*recurrent_zp_folded_bias=*/{0, 0}};
  // positive large weights for input gate to make it really remember
  const GateParameters<float, float, 2, 2> input_gate_params = {
      /*.activation_weight=*/{10, 10, 20, 20},
      /*.recurrent_weight=*/{10, 10, 20, 20},
      /*.fused_bias=*/{-1, -2},
      /*activation_zp_folded_bias=*/{0, 0},
      /*recurrent_zp_folded_bias=*/{0, 0}};
  // all ones to test the behavior of tanh at normal range (-1,1)
  const GateParameters<float, float, 2, 2> cell_gate_params = {
      /*.activation_weight=*/{1, 1, 1, 1},
      /*.recurrent_weight=*/{1, 1, 1, 1},
      /*.fused_bias=*/{0, 0},
      /*activation_zp_folded_bias=*/{0, 0},
      /*recurrent_zp_folded_bias=*/{0, 0}};
  // all ones to test the behavior of sigmoid at normal range (-1. 1)
  const GateParameters<float, float, 2, 2> output_gate_params = {
      /*.activation_weight=*/{1, 1, 1, 1},
      /*.recurrent_weight=*/{1, 1, 1, 1},
      /*.fused_bias=*/{0, 0},
      /*activation_zp_folded_bias=*/{0, 0},
      /*recurrent_zp_folded_bias=*/{0, 0}};

  TestModelContents<float, float, float, float, 2, 3, 2, 2>
      float_model_contents(forget_gate_params, input_gate_params,
                           cell_gate_params, output_gate_params);

  return float_model_contents;
}

ModelQuantizationParameters Get2X2Int8LstmQuantizationSettings() {
  ModelQuantizationParameters quantization_settings;
  quantization_settings.activation_type = kTfLiteInt8;
  quantization_settings.cell_type = kTfLiteInt16;
  quantization_settings.bias_type = kTfLiteInt32;
  quantization_settings.nonlinear_activation_input_scale =
      0.00024414062;  // std::pow(2.0f, -12.0f)
  quantization_settings.nonlinear_activation_output_scale =
      0.00003051757;  // std::pow(2.0f, -15.0f)

  // state quantization parameters
  quantization_settings.input_quantization_parameters = {
      /*scale=*/0.00784313725490196, /*zp=*/0, /*symmetry=*/false};
  quantization_settings.output_quantization_parameters = {
      /*scale=*/0.004705882165580988, /*zp=*/-21, /*symmetry=*/false};
  quantization_settings.hidden_quantization_parameters = {
      /*scale=*/0.004705882165580988, /*zp=*/-21, /*symmetry=*/false};
  quantization_settings.cell_quantization_parameters = {
      /*scale=*/0.00024414062, /*zp=*/0, /*symmetry=*/true};

  // gate quantization parameters
  quantization_settings.forget_gate_quantization_parameters = {
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.0012351397251814111, /*zp=*/0, /*symmetry=*/true}};
  quantization_settings.input_gate_quantization_parameters = {
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.15748031496062992, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.0012351397251814111, /*zp=*/0, /*symmetry=*/true}};
  quantization_settings.cell_gate_quantization_parameters = {
      {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.007874015748031496, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/6.175698625907056e-5, /*zp=*/0, /*symmetry=*/true}};
  quantization_settings.output_gate_quantization_parameters = {
      {/*scale=*/0.1, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.1, /*zp=*/0, /*symmetry=*/true},
      {/*scale=*/0.1, /*zp=*/0, /*symmetry=*/true}};

  return quantization_settings;
}

TestModelContents<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
Create2x3x2X2Int8ModelContents(
    const ModelQuantizationParameters& quantization_settings) {
  auto float_model_contents = Create2x3x2X2FloatModelContents();
  auto quant_model_contents =
      CreateInt8ModelContents(quantization_settings, float_model_contents);
  return quant_model_contents;
}

IntegerLstmParameter CreateIntegerParameter(
    const TfLiteLSTMParams& general_model_settings,
    const ModelQuantizationParameters& quantization_settings) {
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
      quantized_model_contents_.ForgetGateParams().activation_zp_folded_bias;
  evaluation_params.recurrent_to_forget_effective_bias =
      quantized_model_contents_.ForgetGateParams().recurrent_zp_folded_bias;

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
      quantized_model_contents_.InputGateParams().activation_zp_folded_bias;
  evaluation_params.recurrent_to_input_effective_bias =
      quantized_model_contents_.InputGateParams().recurrent_zp_folded_bias;

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
      quantized_model_contents_.CellGateParams().activation_zp_folded_bias;
  evaluation_params.recurrent_to_cell_effective_bias =
      quantized_model_contents_.CellGateParams().recurrent_zp_folded_bias;

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
      quantized_model_contents_.OutputGateParams().activation_zp_folded_bias;
  evaluation_params.recurrent_to_output_effective_bias =
      quantized_model_contents_.OutputGateParams().recurrent_zp_folded_bias;

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
}

}  // namespace testing
}  // namespace tflite