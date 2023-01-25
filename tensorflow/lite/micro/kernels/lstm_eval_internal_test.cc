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
#include "tensorflow/lite/micro/kernels/lstm_eval_internal_test.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval_internal.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)
namespace tflite {
namespace testing {
namespace {
// Test Settings
constexpr float kTestFloatTolerance = 1e-6f;
}  // namespace
}  // namespace testing
}  // namespace tflite
#endif  // !defined(XTENSA)

TF_LITE_MICRO_TESTS_BEGIN
// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)
TF_LITE_MICRO_TEST(CheckGateOutputFloat) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
      float_node_contents = tflite::testing::Create2x3x2X2FloatNodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);
  // Forget gate
  tflite::testing::TestGateOutputFloat<2, 2, 2>(
      float_node_contents.ForgetGateData(), kTfLiteActSigmoid,
      float_node_contents.GetInputData(),
      float_node_contents.GetHiddenStateData(),
      gate_output_data.expected_forget_gate_output,
      tflite::testing::kTestFloatTolerance);
  // Input gate
  tflite::testing::TestGateOutputFloat<2, 2, 2>(
      float_node_contents.InputGateData(), kTfLiteActSigmoid,
      float_node_contents.GetInputData(),
      float_node_contents.GetHiddenStateData(),
      gate_output_data.expected_input_gate_output,
      tflite::testing::kTestFloatTolerance);
  // output gate
  tflite::testing::TestGateOutputFloat<2, 2, 2>(
      float_node_contents.OutputGateData(), kTfLiteActSigmoid,
      float_node_contents.GetInputData(),
      float_node_contents.GetHiddenStateData(),
      gate_output_data.expected_output_gate_output,
      tflite::testing::kTestFloatTolerance);
  // cell (modulation) gate d
  tflite::testing::TestGateOutputFloat<2, 2, 2>(
      float_node_contents.CellGateData(),
      float_node_contents.BuiltinData().activation,
      float_node_contents.GetInputData(),
      float_node_contents.GetHiddenStateData(),
      gate_output_data.expected_cell_gate_output,
      tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckGateOutputInt8) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);
  const tflite::IntegerLstmParameter evaluation_params =
      tflite::testing::CreateIntegerParameter(int8_node_contents);

  // Different gate has different weights, resulting different quantization
  // prediction precisions
  float tolerance;
  // Forget Gate
  // Quantization performs badly here due to integer overflow!!!
  tolerance = 1e-1f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2, 2>(
      int8_node_contents.GetInputData(),
      int8_node_contents.GetHiddenStateData(),
      int8_node_contents.ForgetGateData(),
      int8_node_contents.QuantizationSettings(),
      evaluation_params.effective_input_to_forget_scale_a,
      evaluation_params.effective_input_to_forget_scale_b,
      evaluation_params.effective_recurrent_to_forget_scale_a,
      evaluation_params.effective_recurrent_to_forget_scale_b,
      kTfLiteActSigmoid, gate_output_data.expected_forget_gate_output,
      tolerance);

  // Input Gate
  // Quantization performs badly here due to integer overflow!!!
  tolerance = 1e-1f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2, 2>(
      int8_node_contents.GetInputData(),
      int8_node_contents.GetHiddenStateData(),
      int8_node_contents.InputGateData(),
      int8_node_contents.QuantizationSettings(),
      evaluation_params.effective_input_to_input_scale_a,
      evaluation_params.effective_input_to_input_scale_b,
      evaluation_params.effective_recurrent_to_input_scale_a,
      evaluation_params.effective_recurrent_to_input_scale_b, kTfLiteActSigmoid,
      gate_output_data.expected_input_gate_output, tolerance);

  // Output Gate
  tolerance = 1e-2f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2, 2>(
      int8_node_contents.GetInputData(),
      int8_node_contents.GetHiddenStateData(),
      int8_node_contents.OutputGateData(),
      int8_node_contents.QuantizationSettings(),
      evaluation_params.effective_input_to_output_scale_a,
      evaluation_params.effective_input_to_output_scale_b,
      evaluation_params.effective_recurrent_to_output_scale_a,
      evaluation_params.effective_recurrent_to_output_scale_b,
      kTfLiteActSigmoid, gate_output_data.expected_output_gate_output,
      tolerance);

  // Cell Gate (tanh activation)
  tolerance = 1e-2f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2, 2>(
      int8_node_contents.GetInputData(),
      int8_node_contents.GetHiddenStateData(),
      int8_node_contents.CellGateData(),
      int8_node_contents.QuantizationSettings(),
      evaluation_params.effective_input_to_cell_scale_a,
      evaluation_params.effective_input_to_cell_scale_b,
      evaluation_params.effective_recurrent_to_cell_scale_a,
      evaluation_params.effective_recurrent_to_cell_scale_b,
      int8_node_contents.BuiltinData().activation,
      gate_output_data.expected_cell_gate_output, tolerance);
}

TF_LITE_MICRO_TEST(CheckCellUpdateFloat) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();

  tflite::testing::TestCellUpdateFloat<2, 2, 2>(
      gate_output_data, /*cell_clip=*/6, tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckCellUpdateInt8) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents();
  const tflite::IntegerLstmParameter evaluation_params =
      tflite::testing::CreateIntegerParameter(int8_node_contents);

  // Very high precision. The error is introduced by the
  // quantization error of the clip value (~1e-5), but cannot actually reach
  // the precision due to integer overflow of the elements
  const float tolerance = 1e-3f;
  tflite::testing::TestCellUpdateQuantized<int8_t, int32_t, int16_t, 2, 2, 2>(
      gate_output_data, int8_node_contents.QuantizationSettings(),
      evaluation_params.cell_scale, evaluation_params.quantized_cell_clip,
      tolerance);
}

TF_LITE_MICRO_TEST(CheckOutputCalculationFloat) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  // Not testing projection here, output is the updated hidden state
  tflite::testing::TestHiddenStateUpdateFloat<2, 2, 2>(
      gate_output_data, tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckOutputCalculationInt8) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents();
  const tflite::IntegerLstmParameter evaluation_params =
      tflite::testing::CreateIntegerParameter(int8_node_contents);

  // Theoritical error floor = quantization scale = 0.004705882165580988
  const float tolerance = 1e-2;
  tflite::testing::TestHiddenStateUpdateQuantized<int8_t, int32_t, int16_t, 2,
                                                  2, 2>(
      gate_output_data, int8_node_contents.QuantizationSettings(),
      evaluation_params, tolerance);
}

TF_LITE_MICRO_TEST(CheckOneStepLSTMFloat) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();

  tflite::testing::LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
      float_node_contents = tflite::testing::Create2x3x2X2FloatNodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);
  tflite::testing::TestOneStepLSTMFloat<2, 3, 2, 2>(
      float_node_contents, gate_output_data,
      tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckOneStepLSTMInt8) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);

  const float hidden_state_tolerance = 1e-2;
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-1;
  tflite::testing::TestOneStepLSTMQuantized<int8_t, int32_t, int16_t, 2, 3, 2,
                                            2>(
      int8_node_contents, gate_output_data, hidden_state_tolerance,
      cell_state_tolerance);
}

TF_LITE_MICRO_TEST(TestLSTMEvalFloat) {
  const auto kernel_eval_data = tflite::testing::Get2X2LstmEvalCheckData();

  tflite::testing::LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
      float_node_contents = tflite::testing::Create2x3x2X2FloatNodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);
  tflite::testing::TestLSTMEvalFloat(float_node_contents, kernel_eval_data,
                                     tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(TestLSTMEvalInt8) {
  const auto kernel_eval_data = tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float hidden_state_tolerance = 1e-2;
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-1;
  tflite::testing::TestLSTMEvalQuantized(int8_node_contents, kernel_eval_data,
                                         hidden_state_tolerance,
                                         cell_state_tolerance);
}
#endif  // !defined(XTENSA)
TF_LITE_MICRO_TESTS_END
