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
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval.h"
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
// LSTM internal setting (e.g., nonlinear activation type)
constexpr TfLiteLSTMParams kModelSettings = {
    /*.activation=*/kTfLiteActTanh,
    /*.cell_clip=*/6, /*.proj_clip=*/3,
    /*.kernel_type=*/kTfLiteLSTMFullKernel,
    /*.asymmetric_quantize_inputs=*/true};

/*TEST DATA */
const auto kGateOutputData = Get2X2GateOutputCheckData();
const auto kMultiTimeEvalData = Get2X2LstmEvalCheckData();
const auto kInt8QuantizationSettings = Get2X2Int8LstmQuantizationSettings();

}  // namespace
}  // namespace testing
}  // namespace tflite
#endif  // !defined(XTENSA)

TF_LITE_MICRO_TESTS_BEGIN
// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)
TF_LITE_MICRO_TEST(CheckGateOutputFloat) {
  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      tflite::testing::kGateOutputData.input_data,
      tflite::testing::kGateOutputData.hidden_state,
      tflite::testing::kGateOutputData.cell_state);
  // Forget gate
  tflite::testing::TestGateOutputFloat<2, 2, 2>(
      float_model_contents.ForgetGateParams(), kTfLiteActSigmoid,
      float_model_contents.GetInputData(),
      float_model_contents.GetHiddenStateData(),
      tflite::testing::kGateOutputData.expected_forget_gate_output,
      tflite::testing::kTestFloatTolerance);
  // Input gate
  tflite::testing::TestGateOutputFloat<2, 2, 2>(
      float_model_contents.InputGateParams(), kTfLiteActSigmoid,
      float_model_contents.GetInputData(),
      float_model_contents.GetHiddenStateData(),
      tflite::testing::kGateOutputData.expected_input_gate_output,
      tflite::testing::kTestFloatTolerance);
  // output gate
  tflite::testing::TestGateOutputFloat<2, 2, 2>(
      float_model_contents.OutputGateParams(), kTfLiteActSigmoid,
      float_model_contents.GetInputData(),
      float_model_contents.GetHiddenStateData(),
      tflite::testing::kGateOutputData.expected_output_gate_output,
      tflite::testing::kTestFloatTolerance);
  // cell (modulation) gate d
  tflite::testing::TestGateOutputFloat<2, 2, 2>(
      float_model_contents.CellGateParams(),
      tflite::testing::kModelSettings.activation,
      float_model_contents.GetInputData(),
      float_model_contents.GetHiddenStateData(),
      tflite::testing::kGateOutputData.expected_cell_gate_output,
      tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckGateOutputInt8) {
  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      tflite::testing::kGateOutputData.input_data,
      tflite::testing::kGateOutputData.hidden_state,
      tflite::testing::kGateOutputData.cell_state);
  auto int8_model_contents = tflite::testing::CreateInt8ModelContents(
      tflite::testing::kInt8QuantizationSettings, float_model_contents);
  auto evaluation_params = tflite::testing::CreateIntegerParameter(
      tflite::testing::kModelSettings,
      tflite::testing::kInt8QuantizationSettings, int8_model_contents);

  // Different gate has different weights, resulting different quantization
  // prediction precisions
  float tolerance;
  // Forget Gate
  // Quantization performs badly here due to integer overflow!!!
  tolerance = 1e-1f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2, 2>(
      int8_model_contents.GetInputData(),
      int8_model_contents.GetHiddenStateData(),
      int8_model_contents.ForgetGateParams(),
      tflite::testing::kInt8QuantizationSettings,
      evaluation_params.effective_input_to_forget_scale_a,
      evaluation_params.effective_input_to_forget_scale_b,
      evaluation_params.effective_recurrent_to_forget_scale_a,
      evaluation_params.effective_recurrent_to_forget_scale_b,
      kTfLiteActSigmoid,
      tflite::testing::kGateOutputData.expected_forget_gate_output, tolerance);

  // Input Gate
  // Quantization performs badly here due to integer overflow!!!
  tolerance = 1e-1f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2, 2>(
      int8_model_contents.GetInputData(),
      int8_model_contents.GetHiddenStateData(),
      int8_model_contents.InputGateParams(),
      tflite::testing::kInt8QuantizationSettings,
      evaluation_params.effective_input_to_input_scale_a,
      evaluation_params.effective_input_to_input_scale_b,
      evaluation_params.effective_recurrent_to_input_scale_a,
      evaluation_params.effective_recurrent_to_input_scale_b, kTfLiteActSigmoid,
      tflite::testing::kGateOutputData.expected_input_gate_output, tolerance);

  // Output Gate
  tolerance = 1e-2f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2, 2>(
      int8_model_contents.GetInputData(),
      int8_model_contents.GetHiddenStateData(),
      int8_model_contents.OutputGateParams(),
      tflite::testing::kInt8QuantizationSettings,
      evaluation_params.effective_input_to_output_scale_a,
      evaluation_params.effective_input_to_output_scale_b,
      evaluation_params.effective_recurrent_to_output_scale_a,
      evaluation_params.effective_recurrent_to_output_scale_b,
      kTfLiteActSigmoid,
      tflite::testing::kGateOutputData.expected_output_gate_output, tolerance);

  // Cell Gate (tanh activation)
  tolerance = 1e-2f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2, 2>(
      int8_model_contents.GetInputData(),
      int8_model_contents.GetHiddenStateData(),
      int8_model_contents.CellGateParams(),
      tflite::testing::kInt8QuantizationSettings,
      evaluation_params.effective_input_to_cell_scale_a,
      evaluation_params.effective_input_to_cell_scale_b,
      evaluation_params.effective_recurrent_to_cell_scale_a,
      evaluation_params.effective_recurrent_to_cell_scale_b,
      tflite::testing::kModelSettings.activation,
      tflite::testing::kGateOutputData.expected_cell_gate_output, tolerance);
}

TF_LITE_MICRO_TEST(CheckCellUpdateFloat) {
  tflite::testing::TestCellUpdateFloat<2, 2, 2>(
      tflite::testing::kGateOutputData,
      tflite::testing::kModelSettings.cell_clip,
      tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckCellUpdateInt8) {
  auto float_model_contents =
      tflite::testing::Create2x3x2X2FloatModelContents();
  auto int8_model_contents = tflite::testing::CreateInt8ModelContents(
      tflite::testing::kInt8QuantizationSettings, float_model_contents);
  auto evaluation_params = tflite::testing::CreateIntegerParameter(
      tflite::testing::kModelSettings,
      tflite::testing::kInt8QuantizationSettings, int8_model_contents);

  // Very high precision. The error is introduced by the
  // quantization error of the clip value (~1e-5), but cannot actually reach
  // the precision due to integer overflow of the elements
  const float tolerance = 1e-3f;
  tflite::testing::TestCellUpdateQuantized<int8_t, int32_t, int16_t, 2, 2, 2>(
      tflite::testing::kGateOutputData,
      tflite::testing::kInt8QuantizationSettings, evaluation_params.cell_scale,
      evaluation_params.quantized_cell_clip, tolerance);
}

TF_LITE_MICRO_TEST(CheckOutputCalculationFloat) {
  // Not testing projection here, output is the updated hidden state
  tflite::testing::TestHiddenStateUpdateFloat<2, 2, 2>(
      tflite::testing::kGateOutputData, tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckOutputCalculationInt8) {
  auto float_model_contents =
      tflite::testing::Create2x3x2X2FloatModelContents();
  auto int8_model_contents = tflite::testing::CreateInt8ModelContents(
      tflite::testing::kInt8QuantizationSettings, float_model_contents);
  auto evaluation_params = tflite::testing::CreateIntegerParameter(
      tflite::testing::kModelSettings,
      tflite::testing::kInt8QuantizationSettings, int8_model_contents);

  // Theoritical error floor = quantization scale = 0.004705882165580988
  const float tolerance = 1e-2;
  tflite::testing::TestHiddenStateUpdateQuantized<int8_t, int32_t, int16_t, 2,
                                                  2, 2>(
      tflite::testing::kGateOutputData,
      tflite::testing::kInt8QuantizationSettings, evaluation_params, tolerance);
}

TF_LITE_MICRO_TEST(CheckOneStepLSTMFloat) {
  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      tflite::testing::kGateOutputData.input_data,
      tflite::testing::kGateOutputData.hidden_state,
      tflite::testing::kGateOutputData.cell_state);
  tflite::testing::TestOneStepLSTMFloat<2, 3, 2, 2>(
      tflite::testing::kModelSettings, float_model_contents,
      tflite::testing::kGateOutputData, tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(CheckOneStepLSTMInt8) {
  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      tflite::testing::kGateOutputData.input_data,
      tflite::testing::kGateOutputData.hidden_state,
      tflite::testing::kGateOutputData.cell_state);
  auto int8_model_contents = tflite::testing::CreateInt8ModelContents(
      tflite::testing::kInt8QuantizationSettings, float_model_contents);
  auto evaluation_params = tflite::testing::CreateIntegerParameter(
      tflite::testing::kModelSettings,
      tflite::testing::kInt8QuantizationSettings, int8_model_contents);

  const float hidden_state_tolerance = 1e-2;
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-1;
  tflite::testing::TestOneStepLSTMQuantized<int8_t, int32_t, int16_t, 2, 3, 2,
                                            2>(
      int8_model_contents, tflite::testing::kInt8QuantizationSettings,
      evaluation_params, tflite::testing::kGateOutputData,
      hidden_state_tolerance, cell_state_tolerance);
}

TF_LITE_MICRO_TEST(TestLSTMEvalFloat) {
  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      tflite::testing::kMultiTimeEvalData.input_data,
      tflite::testing::kMultiTimeEvalData.hidden_state);
  tflite::testing::TestLSTMEvalFloat(tflite::testing::kModelSettings,
                                     float_model_contents,
                                     tflite::testing::kMultiTimeEvalData,
                                     tflite::testing::kTestFloatTolerance);
}

TF_LITE_MICRO_TEST(TestLSTMEvalInt8) {
  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      tflite::testing::kMultiTimeEvalData.input_data,
      tflite::testing::kMultiTimeEvalData.hidden_state);
  auto int8_model_contents = tflite::testing::CreateInt8ModelContents(
      tflite::testing::kInt8QuantizationSettings, float_model_contents);
  auto evaluation_params = tflite::testing::CreateIntegerParameter(
      tflite::testing::kModelSettings,
      tflite::testing::kInt8QuantizationSettings, int8_model_contents);

  const float hidden_state_tolerance = 1e-2;
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-1;
  tflite::testing::TestLSTMEvalQuantized(
      tflite::testing::kModelSettings, int8_model_contents,
      tflite::testing::kInt8QuantizationSettings, evaluation_params,
      tflite::testing::kMultiTimeEvalData, hidden_state_tolerance,
      cell_state_tolerance);
}
#endif  // !defined(XTENSA)
TF_LITE_MICRO_TESTS_END
