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
#include "tensorflow/lite/micro/kernels/lstm_eval.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {
constexpr int kInputDimension = 2;
constexpr int kStateDimension = 2;
constexpr int kBatchSize = 1;

constexpr int kGateOutputSize = kBatchSize * kStateDimension;

constexpr TfLiteLSTMParams kModelSettings = {
    /*.activation=*/kTfLiteActTanh,
    /*.cell_clip=*/10, /*.proj_clip=*/3, /*.kernel_type=*/kTfLiteLSTMFullKernel,
    /*.asymmetric_quantize_inputs=*/true};

// Struct that holds the weight/bias information for a standard gate (i.e. no
// modification such as layer normalization, peephole, etc.)
struct GateParameters {
  const float activation_weight[kInputDimension * kStateDimension];
  const float recurrent_weight[kInputDimension * kStateDimension];
  const float fused_bias[kStateDimension];
};

// Set parameters for different gates
// negative large weights for forget gate to make it really forget
constexpr GateParameters kForgetGateParameters = {
    /*.activation_weight=*/{-10, -10, -20, -20},
    /*.recurrent_weight=*/{-10, -10, -20, -20},
    /*.fused_bias=*/{1, 2}};
// positive large weights for input gate to make it really remember
constexpr GateParameters kInputGateParameters = {
    /*.activation_weight=*/{10, 10, 20, 20},
    /*.recurrent_weight=*/{10, 10, 20, 20},
    /*.fused_bias=*/{-1, -2}};
// all ones to test the behavior of tanh at normal range (-1,1)
constexpr GateParameters kModulationGateParameters = {
    /*.activation_weight=*/{1, 1, 1, 1},
    /*.recurrent_weight=*/{1, 1, 1, 1},
    /*.fused_bias=*/{0, 0}};
// all ones to test the behavior of sigmoid at normal range (-1. 1)
constexpr GateParameters kOutputGateParameters = {
    /*.activation_weight=*/{1, 1, 1, 1},
    /*.recurrent_weight=*/{1, 1, 1, 1},
    /*.fused_bias=*/{0, 0}};

template <typename T>
void ValidateResultGoldens(const T* golden, const T* output_data,
                           const int output_len, const float tolerance) {
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

void TestGateOutputFloat(const GateParameters& gate_params,
                         TfLiteFusedActivation activation_type,
                         const float* input_dats, const float* recurrent_state,
                         const float* expected_vals) {
  float gate_output[kGateOutputSize];
  tflite::lstm_internal::CalculateLstmGateFloat(
      input_dats, gate_params.activation_weight,
      /*aux_input=*/nullptr, /*aux_input_to_gate_weights*/ nullptr,
      recurrent_state, gate_params.recurrent_weight,
      /*cell_state=*/nullptr, /*cell_to_gate_weights=*/nullptr,
      /*layer_norm_coefficients=*/nullptr, gate_params.fused_bias, kBatchSize,
      kInputDimension, kInputDimension, kStateDimension, kStateDimension,
      /*activation=*/activation_type, gate_output,
      /*is_input_all_zeros=*/false,
      /*is_aux_input_all_zeros=*/true);
  ValidateResultGoldens(expected_vals, gate_output, kGateOutputSize, 1e-8f);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN
TF_LITE_MICRO_TEST(CheckGateOutputFloat) {
  const float input_data[] = {0.2, 0.3};
  const float recurrent_state[] = {-0.1, 0.2};
  // Use the forget gate parameters to test small gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[-10,-10],[-20,-20]][0.2,
  // +[[-10,-10],[-20,-20]][-0.1, 0.2]+[1,2]) = sigmoid([-5,-10]) =
  // [6.69285092e-03, 4.53978687e-05]
  const float expected_forget_gate_output[] = {6.69285092e-3f, 4.53978687e-5f};
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kForgetGateParameters, kTfLiteActSigmoid, input_data,
      recurrent_state, expected_forget_gate_output);

  // Use the input gate parameters to test small gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[10,10],[20,20]][0.2, 0.3]
  // +[[10,10],[20,20]][-0.1, 0.2]+[-1,-2]) = sigmoid([5,10]) =
  // [0.99330715, 0.9999546]
  const float expected_input_gate_output[] = {0.99330715, 0.9999546};
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kInputGateParameters, kTfLiteActSigmoid, input_data,
      recurrent_state, expected_input_gate_output);

  // Use the output gate parameters to test normnal gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[1,1],[1,1]][0.2, 0.3]
  // +[[1,1],[1,1]][-0.1, 0.2]+[0,0]) = sigmoid([0.6,0.6]) =
  // [0.6456563062257954, 0.6456563062257954]
  const float expected_output_gate_output[] = {0.6456563062257954,
                                               0.6456563062257954};
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kOutputGateParameters, kTfLiteActSigmoid, input_data,
      recurrent_state, expected_output_gate_output);

  // Use the modulation(cell) gate parameters to tanh output
  // output = tanh(W_i*i+W_h*h+b) = tanh([[1,1],[1,1]][0.2, 0.3]
  // +[[1,1],[1,1]][-0.1, 0.2]+[0,0]) = tanh([0.6,0.6]) =
  // [0.6456563062257954, 0.6456563062257954]
  const float expected_modulation_gate_output[] = {0.5370495669980353,
                                                   0.5370495669980353};
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kModulationGateParameters,
      tflite::testing::kModelSettings.activation, input_data, recurrent_state,
      expected_modulation_gate_output);
}

TF_LITE_MICRO_TEST(CheckCellUpdateFloat) {
  float cell_state[] = {0.1, 0.2};
  float forget_gate[] = {0.2, 0.5};
  const float input_gate[] = {0.8, 0.9};
  const float cell_gate[] = {-0.3, 0.8};  // modulation gate

  // Cell = forget_gate*cell + input_gate*modulation_gate
  // = [0.02, 0.1] + [-0.24, 0.72] = [-0.22, 0.82]
  const float expected_cell_vals[] = {-0.22, 0.82};

  tflite::lstm_internal::UpdateLstmCellFloat(
      tflite::testing::kBatchSize, tflite::testing::kStateDimension, cell_state,
      input_gate, forget_gate, cell_gate, /*use_cifg=*/false,
      /*clip=*/tflite::testing::kModelSettings.cell_clip);

  tflite::testing::ValidateResultGoldens(
      expected_cell_vals, cell_state, tflite::testing::kGateOutputSize, 1e-6f);
}

TF_LITE_MICRO_TEST(CheckOutputCalculationFloat) {
  // -1 and 5 for different tanh behavior (normal, saturated)
  const float cell_state[] = {-1, 5};
  const float output_gate[] = {0.2, 0.5};
  // If no projection layer, hidden state dimension == output dimension == cell
  // state dimension
  float output[tflite::testing::kGateOutputSize];
  float scratch[tflite::testing::kGateOutputSize];

  tflite::lstm_internal::CalculateLstmOutputFloat(
      tflite::testing::kBatchSize, tflite::testing::kStateDimension,
      tflite::testing::kStateDimension, cell_state, output_gate, kTfLiteActTanh,
      nullptr, nullptr, 0, output, scratch);

  // Output state generate the output and copy it to the hidden state
  // tanh(cell_state) * output_gate =
  // [-0.7615941559557649,0.9999092042625951] * [0.2, 0.5] =
  // [-0.15231883119115297, 0.49995460213129755]
  float expected_output_vals[] = {-0.15231883119115297, 0.49995460213129755};
  tflite::testing::ValidateResultGoldens(
      expected_output_vals, output, tflite::testing::kGateOutputSize, 1e-6f);
}

// TF_LITE_MICRO_TEST(CheckOneStepLSTMFloat) {
//   const float input_data[] = {0.2, 0.3};
//   float recurrent_state[tflite::testing::kGateOutputSize];
//   float cell_state[tflite::testing::kGateOutputSize];
//   float output[tflite::testing::kGateOutputSize];

//   float cell_gate_scratch[tflite::testing::kGateOutputSize];
//   float forget_gate_scratch[tflite::testing::kGateOutputSize];
//   float input_gate_scratch[tflite::testing::kGateOutputSize];
//   float output_gate_scratch[tflite::testing::kGateOutputSize];

//   LstmStepFloat(
//       input_data, tflite::testing::kInputGateParameters.activation_weight,
//       tflite::testing::kForgetGateParameters.activation_weight,
//       tflite::testing::kModulationGateParameters.activation_weight,
//       tflite::testing::kOutputGateParameters.activation_weight,
//       /*aux_input_ptr=*/nullptr, /*aux_input_to_input_weights_ptr=*/nullptr,
//       /*aux_input_to_forget_weights_ptr=*/nullptr,
//       /*aux_input_to_cell_weights_ptr=*/nullptr,
//       /*aux_input_to_output_weights_ptr=*/nullptr,
//       tflite::testing::kInputGateParameters.recurrent_weight,
//       tflite::testing::kForgetGateParameters.recurrent_weight,
//       tflite::testing::kModulationGateParameters.recurrent_weight,
//       tflite::testing::kOutputGateParameters.recurrent_weight,
//       /*cell_to_input_weights_ptr=*/nullptr,
//       /*cell_to_forget_weights_ptr=*/nullptr,
//       /*cell_to_output_weights_ptr=*/nullptr,
//       /*input_layer_norm_coefficients_ptr=*/nullptr,
//       /*forget_layer_norm_coefficients_ptr=*/nullptr,
//       /*cell_layer_norm_coefficients_ptr=*/nullptr,
//       /*output_layer_norm_coefficients_ptr=*/nullptr,
//       tflite::testing::kInputGateParameters.fused_bias,
//       tflite::testing::kForgetGateParameters.fused_bias,
//       tflite::testing::kModulationGateParameters.fused_bias,
//       tflite::testing::kOutputGateParameters.fused_bias, ,
//       /*projection_weights_ptr=*/nullptr, /*projection_bias_ptr=*/nullptr,
//       const TfLiteLSTMParams* params, tflite::testing::kBatchSize,
//       tflite::testing::kStateDimension, tflite::testing::kInputDimension,
//       tflite::testing::kInputDimension, tflite::testing::kStateDimension,
//       /*output_batch_leading_dim=*/0, recurrent_state, cell_state,
//       input_gate_scratch forget_gate_scratch, cell_gate_scratch,
//       output_gate_scratch, output)
// }
TF_LITE_MICRO_TESTS_END