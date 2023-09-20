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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)
namespace {
// Test Settings
constexpr float kTestFloatTolerance = 1e-6f;
}  // namespace
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
  tflite::testing::TestCalculateLstmGateFloat<2, 2>(
      float_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      float_node_contents.GetEvalTensor(
          tflite::kLstmInputToForgetWeightsTensor),
      float_node_contents.GetEvalTensor(tflite::kLstmForgetGateBiasTensor),
      // Recurrent FC
      float_node_contents.HiddenStateEvalTensor(),
      float_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToForgetWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_forget_gate_output,
      kTestFloatTolerance);

  // Input gate
  tflite::testing::TestCalculateLstmGateFloat<2, 2>(
      float_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      float_node_contents.GetEvalTensor(tflite::kLstmInputToInputWeightsTensor),
      float_node_contents.GetEvalTensor(tflite::kLstmInputGateBiasTensor),
      // Recurrent FC
      float_node_contents.HiddenStateEvalTensor(),
      float_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToInputWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_input_gate_output,
      kTestFloatTolerance);

  // Output gate
  tflite::testing::TestCalculateLstmGateFloat<2, 2>(
      float_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      float_node_contents.GetEvalTensor(
          tflite::kLstmInputToOutputWeightsTensor),
      float_node_contents.GetEvalTensor(tflite::kLstmOutputGateBiasTensor),
      // Recurrent FC
      float_node_contents.HiddenStateEvalTensor(),
      float_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToOutputWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_output_gate_output,
      kTestFloatTolerance);

  // Cell gate
  tflite::testing::TestCalculateLstmGateFloat<2, 2>(
      float_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      float_node_contents.GetEvalTensor(tflite::kLstmInputToCellWeightsTensor),
      float_node_contents.GetEvalTensor(tflite::kLstmCellGateBiasTensor),
      // Recurrent FC
      float_node_contents.HiddenStateEvalTensor(),
      float_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToCellWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Result comparison
      float_node_contents.BuiltinData().activation,
      gate_output_data.expected_cell_gate_output, kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckGateOutputInt8) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);

  // Forget gate
  // Quantization performs badly here due to integer overflow!!!
  float tolerance = 1e-1f;
  tflite::testing::TestCalculateLstmGateInteger<int8_t, int8_t, int32_t,
                                                int16_t, 2, 2>(
      int8_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      int8_node_contents.GetEvalTensor(tflite::kLstmInputToForgetWeightsTensor),
      int8_node_contents.GetEvalTensor(tflite::kLstmForgetGateBiasTensor),
      // Recurrent FC
      int8_node_contents.HiddenStateEvalTensor(),
      int8_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToForgetWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      int8_node_contents.QuantizationSettings(),
      int8_node_contents.QuantizationSettings().forget_gate,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_forget_gate_output,
      tolerance);

  // Input gate
  // Quantization performs badly here due to integer overflow!!!
  tolerance = 1e-1f;
  tflite::testing::TestCalculateLstmGateInteger<int8_t, int8_t, int32_t,
                                                int16_t, 2, 2>(
      int8_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      int8_node_contents.GetEvalTensor(tflite::kLstmInputToInputWeightsTensor),
      int8_node_contents.GetEvalTensor(tflite::kLstmInputGateBiasTensor),
      // Recurrent FC
      int8_node_contents.HiddenStateEvalTensor(),
      int8_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToInputWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      int8_node_contents.QuantizationSettings(),
      int8_node_contents.QuantizationSettings().input_gate,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_input_gate_output,
      tolerance);

  // Output gate
  tolerance = 1e-2f;
  tflite::testing::TestCalculateLstmGateInteger<int8_t, int8_t, int32_t,
                                                int16_t, 2, 2>(
      int8_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      int8_node_contents.GetEvalTensor(tflite::kLstmInputToOutputWeightsTensor),
      int8_node_contents.GetEvalTensor(tflite::kLstmOutputGateBiasTensor),
      // Recurrent FC
      int8_node_contents.HiddenStateEvalTensor(),
      int8_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToOutputWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      int8_node_contents.QuantizationSettings(),
      int8_node_contents.QuantizationSettings().output_gate,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_output_gate_output,
      tolerance);

  // Cell gate
  tolerance = 1e-2f;
  tflite::testing::TestCalculateLstmGateInteger<int8_t, int8_t, int32_t,
                                                int16_t, 2, 2>(
      int8_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      int8_node_contents.GetEvalTensor(tflite::kLstmInputToCellWeightsTensor),
      int8_node_contents.GetEvalTensor(tflite::kLstmCellGateBiasTensor),
      // Recurrent FC
      int8_node_contents.HiddenStateEvalTensor(),
      int8_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToCellWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      int8_node_contents.QuantizationSettings(),
      int8_node_contents.QuantizationSettings().cell_gate,
      // Result comparison
      int8_node_contents.BuiltinData().activation,
      gate_output_data.expected_cell_gate_output, tolerance);
}

TF_LITE_MICRO_TEST(CheckGateOutputInt16) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);

  // Forget gate
  // Quantization performs badly here due to integer overflow (from batch2)!!!
  float tolerance = 1e-1f;
  tflite::testing::TestCalculateLstmGateInteger<int16_t, int8_t, int64_t,
                                                int16_t, 2, 2>(
      int16_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      int16_node_contents.GetEvalTensor(
          tflite::kLstmInputToForgetWeightsTensor),
      int16_node_contents.GetEvalTensor(tflite::kLstmForgetGateBiasTensor),
      // Recurrent FC
      int16_node_contents.HiddenStateEvalTensor(),
      int16_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToForgetWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      int16_node_contents.QuantizationSettings(),
      int16_node_contents.QuantizationSettings().forget_gate,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_forget_gate_output,
      tolerance);

  // Input gate
  // Quantization performs badly here due to integer overflow (from batch2)!!!
  tolerance = 1e-1f;
  tflite::testing::TestCalculateLstmGateInteger<int16_t, int8_t, int64_t,
                                                int16_t, 2, 2>(
      int16_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      int16_node_contents.GetEvalTensor(tflite::kLstmInputToInputWeightsTensor),
      int16_node_contents.GetEvalTensor(tflite::kLstmInputGateBiasTensor),
      // Recurrent FC
      int16_node_contents.HiddenStateEvalTensor(),
      int16_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToInputWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      int16_node_contents.QuantizationSettings(),
      int16_node_contents.QuantizationSettings().input_gate,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_input_gate_output,
      tolerance);

  // Output gate
  // Quantization scale (theoritical lowest range) is at range 1e-5
  tolerance = 1e-4f;
  tflite::testing::TestCalculateLstmGateInteger<int16_t, int8_t, int64_t,
                                                int16_t, 2, 2>(
      int16_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      int16_node_contents.GetEvalTensor(
          tflite::kLstmInputToOutputWeightsTensor),
      int16_node_contents.GetEvalTensor(tflite::kLstmOutputGateBiasTensor),
      // Recurrent FC
      int16_node_contents.HiddenStateEvalTensor(),
      int16_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToOutputWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      int16_node_contents.QuantizationSettings(),
      int16_node_contents.QuantizationSettings().output_gate,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_output_gate_output,
      tolerance);

  // Cell gate
  tolerance = 1e-4f;
  tflite::testing::TestCalculateLstmGateInteger<int16_t, int8_t, int64_t,
                                                int16_t, 2, 2>(
      int16_node_contents.GetEvalTensor(tflite::kLstmInputTensor),
      int16_node_contents.GetEvalTensor(tflite::kLstmInputToCellWeightsTensor),
      int16_node_contents.GetEvalTensor(tflite::kLstmCellGateBiasTensor),
      // Recurrent FC
      int16_node_contents.HiddenStateEvalTensor(),
      int16_node_contents.GetEvalTensor(
          tflite::kLstmRecurrentToCellWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      int16_node_contents.QuantizationSettings(),
      int16_node_contents.QuantizationSettings().cell_gate,
      // Result comparison
      int16_node_contents.BuiltinData().activation,
      gate_output_data.expected_cell_gate_output, tolerance);
}

TF_LITE_MICRO_TEST(CheckCellStateUpdateFloat) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
      float_node_contents = tflite::testing::Create2x3x2X2FloatNodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);

  tflite::testing::TestUpdateLstmCellFloat(
      gate_output_data, float_node_contents, kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckCellStateUpdateInt8) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);

  // Very high precision. The error is introduced by the
  // quantization error of the clip value (~1e-5), but cannot actually reach
  // the precision due to integer overflow of the elements
  const float tolerance = 1e-3f;
  tflite::testing::TestUpdateLstmCellInteger(gate_output_data,
                                             int8_node_contents, tolerance);
}

TF_LITE_MICRO_TEST(CheckCellStateUpdateInt16) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);
  // Very high precision. The error is introduced by the
  // quantization error of the clip value (~1e-5), but cannot actually reach
  // the precision due to integer overflow of the elements
  const float tolerance = 1e-3f;
  tflite::testing::TestUpdateLstmCellInteger(gate_output_data,
                                             int16_node_contents, tolerance);
}

TF_LITE_MICRO_TEST(CheckHiddenStateUpdateFloat) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
      float_node_contents = tflite::testing::Create2x3x2X2FloatNodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.expected_updated_cell);

  tflite::testing::TestUpdateLstmHiddenFloat(
      gate_output_data, float_node_contents, kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckHiddenStateUpdateInt8) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.expected_updated_cell);

  // Theoritical error floor = quantization scale = 0.004705882165580988
  const float tolerance = 1e-2;
  tflite::testing::TestUpdateLstmHiddenInteger(gate_output_data,
                                               int8_node_contents, tolerance);
}

TF_LITE_MICRO_TEST(CheckHiddenStateUpdateInt16) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.expected_updated_cell);

  const float tolerance = 1e-4;
  tflite::testing::TestUpdateLstmHiddenInteger(gate_output_data,
                                               int16_node_contents, tolerance);
}

TF_LITE_MICRO_TEST(CheckOneStepLSTMFloat) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
      float_node_contents = tflite::testing::Create2x3x2X2FloatNodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);
  tflite::testing::TestLstmStepFloat(gate_output_data, kTestFloatTolerance,
                                     kTestFloatTolerance, float_node_contents);
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
  tflite::testing::TestLstmStepInteger(gate_output_data, hidden_state_tolerance,
                                       cell_state_tolerance,
                                       int8_node_contents);
}

TF_LITE_MICRO_TEST(CheckOneStepLSTMInt16) {
  const tflite::testing::GateOutputCheckData<4, 4> gate_output_data =
      tflite::testing::Get2X2GateOutputCheckData();
  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          gate_output_data.input_data, gate_output_data.hidden_state,
          gate_output_data.cell_state);
  const float hidden_state_tolerance = 1e-3;  // actually very close to 1e-4
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-1;
  tflite::testing::TestLstmStepInteger<int16_t, int8_t, int64_t, int16_t, 2, 3,
                                       2, 2>(
      gate_output_data, hidden_state_tolerance, cell_state_tolerance,
      int16_node_contents);
}

TF_LITE_MICRO_TEST(TestLSTMEvalFloat) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
      float_node_contents = tflite::testing::Create2x3x2X2FloatNodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  tflite::testing::TestEvalLstmFloat(kernel_eval_data, kTestFloatTolerance,
                                     kTestFloatTolerance, float_node_contents);
}

TF_LITE_MICRO_TEST(TestLSTMEvalInt8) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float hidden_state_tolerance = 1e-2;
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-2;
  tflite::testing::TestEvalLstmInteger(kernel_eval_data, hidden_state_tolerance,
                                       cell_state_tolerance,
                                       int8_node_contents);
}

TF_LITE_MICRO_TEST(TestLSTMEvalInt16) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float hidden_state_tolerance = 1e-3;  // actually very close to 1e-4
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-2;
  tflite::testing::TestEvalLstmInteger(kernel_eval_data, hidden_state_tolerance,
                                       cell_state_tolerance,
                                       int16_node_contents);
}
#endif  // !defined(XTENSA)

TF_LITE_MICRO_TESTS_END
