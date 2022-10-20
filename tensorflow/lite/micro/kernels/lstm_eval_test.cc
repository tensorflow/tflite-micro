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

// Struct that holds the weight/bias information for a standard gate (i.e. no
// modification such as layer normalization, peephole, etc.)
struct GateParameters {
  const float activation_weight[kInputDimension * kStateDimension * kBatchSize];
  const float recurrent_weight[kInputDimension * kStateDimension * kBatchSize];
  const float fused_bias[kStateDimension * kBatchSize];
};

// Set parameters for different gates
// negative large weights for forget gate to make it really forget
const GateParameters kForgetGateParameters = {
    /*.activation_weight=*/{-10, -10, -20, -20},
    /*.recurrent_weight=*/{-10, -10, -20, -20},
    /*.fused_bias=*/{1, 2}};
// positive large weights for input gate to make it really remember
const GateParameters kInputGateParameters = {
    /*.activation_weight=*/{10, 10, 20, 20},
    /*.recurrent_weight=*/{10, 10, 20, 20},
    /*.fused_bias=*/{-1, -2}};
// all ones to test the behavior of tanh at normal range (-1,1)
const GateParameters kModulationGateParameters = {
    /*.activation_weight=*/{1, 1, 1, 1},
    /*.recurrent_weight=*/{1, 1, 1, 1},
    /*.fused_bias=*/{0, 0}};
// all ones to test the behavior of sigmoid at normal range (-1. 1)
const GateParameters kOutputGateParameters = {
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
                         const float* input_dats, const float* recurrent_data,
                         const float* expected_vals) {
  const int output_size = kBatchSize * kStateDimension;
  float gate_output[output_size];
  tflite::lstm_internal::CalculateLstmGateFloat(
      input_dats, gate_params.activation_weight,
      /*aux_input=*/nullptr, /*aux_input_to_gate_weights*/ nullptr,
      recurrent_data, gate_params.recurrent_weight,
      /*cell_state=*/nullptr, /*cell_to_gate_weights=*/nullptr,
      /*layer_norm_coefficients=*/nullptr, gate_params.fused_bias, kBatchSize,
      kInputDimension, kInputDimension, kStateDimension, kStateDimension,
      /*activation=*/activation_type, gate_output,
      /*is_input_all_zeros=*/false,
      /*is_aux_input_all_zeros=*/true);
  ValidateResultGoldens(expected_vals, gate_output, output_size, 1e-8f);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN
TF_LITE_MICRO_TEST(CheckGateOutputFloat) {
  const float input_data[] = {0.2, 0.3};
  const float recurrent_data[] = {-0.1, 0.2};
  // Use the forget gate parameters to test small gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[-10,-10],[-20,-20]][0.2,
  // +[[-10,-10],[-20,-20]][-0.1, 0.2]+[1,2]) = sigmoid([-5,-10]) =
  // [6.69285092e-03, 4.53978687e-05]
  const float expected_forget_gate_output[] = {6.69285092e-3f, 4.53978687e-5f};
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kForgetGateParameters, kTfLiteActSigmoid, input_data,
      recurrent_data, expected_forget_gate_output);

  // Use the input gate parameters to test small gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[10,10],[20,20]][0.2, 0.3]
  // +[[10,10],[20,20]][-0.1, 0.2]+[-1,-2]) = sigmoid([5,10]) =
  // [0.99330715, 0.9999546]
  const float expected_input_gate_output[] = {0.99330715, 0.9999546};
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kInputGateParameters, kTfLiteActSigmoid, input_data,
      recurrent_data, expected_input_gate_output);

  // Use the output gate parameters to test normnal gate outputs
  // output = sigmoid(W_i*i+W_h*h+b) = sigmoid([[1,1],[1,1]][0.2, 0.3]
  // +[[1,1],[1,1]][-0.1, 0.2]+[0,0]) = sigmoid([0.6,0.6]) =
  // [0.6456563062257954, 0.6456563062257954]
  const float expected_output_gate_output[] = {0.6456563062257954,
                                               0.6456563062257954};
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kOutputGateParameters, kTfLiteActSigmoid, input_data,
      recurrent_data, expected_output_gate_output);

  // Use the modulation(cell) gate parameters to tanh output
  // output = tanh(W_i*i+W_h*h+b) = tanh([[1,1],[1,1]][0.2, 0.3]
  // +[[1,1],[1,1]][-0.1, 0.2]+[0,0]) = tanh([0.6,0.6]) =
  // [0.6456563062257954, 0.6456563062257954]
  const float expected_modulation_gate_output[] = {0.5370495669980353,
                                                   0.5370495669980353};
  tflite::testing::TestGateOutputFloat(
      tflite::testing::kModulationGateParameters, kTfLiteActTanh, input_data,
      recurrent_data, expected_modulation_gate_output);
}

TF_LITE_MICRO_TEST(CheckCellUpdateFloat) {
  float cell_data[] = {0.1, 0.2};
  float forget_gate[] = {0.2, 0.5};
  const float input_gate[] = {0.8, 0.9};
  const float cell_gate[] = {-0.3, 0.8};  // modulation gate
  const int cell_size =
      tflite::testing::kBatchSize * tflite::testing::kStateDimension;

  // Cell = forget_gate*cell + input_gate*modulation_gate
  // = [0.02, 0.1] + [-0.24, 0.72] = [-0.22, 0.82]
  const float expected_cell_vals[] = {-0.22, 0.82};

  tflite::lstm_internal::UpdateLstmCellFloat(
      tflite::testing::kBatchSize, tflite::testing::kStateDimension, cell_data,
      input_gate, forget_gate, cell_gate, /*use_cifg=*/false, /*clip=*/10);

  tflite::testing::ValidateResultGoldens(expected_cell_vals, cell_data,
                                         cell_size, 1e-6f);
}

TF_LITE_MICRO_TEST(CheckOutputFloat) {
  const float cell_data[] = {-1, 5};
  const float output_gate[] = {0.2, 0.5};
  const int output_size =
      tflite::testing::kStateDimension * tflite::testing::kBatchSize;
  float output[output_size];
  float scratch[output_size];

  tflite::lstm_internal::CalculateLstmOutputFloat(
      tflite::testing::kBatchSize, tflite::testing::kStateDimension,
      tflite::testing::kStateDimension, cell_data, output_gate, kTfLiteActTanh,
      nullptr, nullptr, 0, output, scratch);

  // Output state generate the output and copy it to the hidden state
  // tanh(cell_state) * output_gate =
  // [-0.7615941559557649,0.9999092042625951] * [0.2, 0.5] =
  // [-0.15231883119115297, 0.49995460213129755]
  float expected_output_vals[] = {-0.15231883119115297, 0.49995460213129755};
  tflite::testing::ValidateResultGoldens(expected_output_vals, output,
                                         output_size, 1e-6f);
}

TF_LITE_MICRO_TESTS_END