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
#include "tensorflow/lite/micro/kernels/lstm_eval_16act_test.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/lstm_eval_16act.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
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
constexpr TfLiteUnidirectionalSequenceLSTMParams kModelSettings = {
    /*.activation=*/kTfLiteActTanh,
    /*.cell_clip=*/6, /*.proj_clip=*/3,
    /*.time_major=*/false,
    /*.asymmetric_quantize_inputs=*/true};

}  // namespace
}  // namespace testing
}  // namespace tflite
#endif  // !defined(XTENSA)

TF_LITE_MICRO_TESTS_BEGIN
// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)
TF_LITE_MICRO_TEST(CheckGateOutputInt8) {
  const auto gate_output_data = tflite::testing::Get2X2GateOutputCheckData();
  const auto quantization_settings =
      tflite::testing::Get2X2Int8LstmQuantizationSettings();

  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      gate_output_data.input_data, gate_output_data.hidden_state,
      gate_output_data.cell_state);
  auto int8_model_contents = tflite::testing::CreateInt8ModelContents(
      quantization_settings, float_model_contents);

  // get step information: only one time step, no need to update
  // set time_major = true to test batch inference
  auto size_info = tflite::testing::CreateLstmSizeInfo(
      /*time_major*/ true,
      int8_model_contents.GetInternalTensor(tflite::kLstmInputTensor)->dims,
      int8_model_contents.HiddenStateTensor()->dims);
  tflite::lstm_internal::LstmStepManager step_info(size_info);

  // Forget gate
  // Quantization performs badly here due to integer overflow!!!
  float tolerance = 1e-1f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2>(
      step_info,
      int8_model_contents.GetInternalTensor(tflite::kLstmInputTensor),
      int8_model_contents.GetInternalTensor(
          tflite::kLstmInputToForgetWeightsTensor),
      int8_model_contents.GetInternalTensor(tflite::kLstmForgetGateBiasTensor),
      // Recurrent FC
      int8_model_contents.HiddenStateTensor(),
      int8_model_contents.GetInternalTensor(
          tflite::kLstmRecurrentToForgetWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      quantization_settings,
      quantization_settings.forget_gate_quantization_parameters,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_forget_gate_output,
      tolerance);

  // Input gate
  // Quantization performs badly here due to integer overflow!!!
  tolerance = 1e-1f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2>(
      step_info,
      int8_model_contents.GetInternalTensor(tflite::kLstmInputTensor),
      int8_model_contents.GetInternalTensor(
          tflite::kLstmInputToInputWeightsTensor),
      int8_model_contents.GetInternalTensor(tflite::kLstmInputGateBiasTensor),
      // Recurrent FC
      int8_model_contents.HiddenStateTensor(),
      int8_model_contents.GetInternalTensor(
          tflite::kLstmRecurrentToInputWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      quantization_settings,
      quantization_settings.input_gate_quantization_parameters,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_input_gate_output,
      tolerance);

  // Output gate
  tolerance = 1e-2f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2>(
      step_info,
      int8_model_contents.GetInternalTensor(tflite::kLstmInputTensor),
      int8_model_contents.GetInternalTensor(
          tflite::kLstmInputToOutputWeightsTensor),
      int8_model_contents.GetInternalTensor(tflite::kLstmOutputGateBiasTensor),
      // Recurrent FC
      int8_model_contents.HiddenStateTensor(),
      int8_model_contents.GetInternalTensor(
          tflite::kLstmRecurrentToOutputWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      quantization_settings,
      quantization_settings.output_gate_quantization_parameters,
      // Result comparison
      kTfLiteActSigmoid, gate_output_data.expected_output_gate_output,
      tolerance);

  // Cell gate
  tolerance = 1e-2f;
  tflite::testing::TestGateOutputQuantized<int8_t, int32_t, int16_t, 2, 2>(
      step_info,
      int8_model_contents.GetInternalTensor(tflite::kLstmInputTensor),
      int8_model_contents.GetInternalTensor(
          tflite::kLstmInputToCellWeightsTensor),
      int8_model_contents.GetInternalTensor(tflite::kLstmCellGateBiasTensor),
      // Recurrent FC
      int8_model_contents.HiddenStateTensor(),
      int8_model_contents.GetInternalTensor(
          tflite::kLstmRecurrentToCellWeightsTensor),
      nullptr,  // bias fused to activation FC,
      // Quantization settings
      quantization_settings,
      quantization_settings.cell_gate_quantization_parameters,
      // Result comparison
      tflite::testing::kModelSettings.activation,
      gate_output_data.expected_cell_gate_output, tolerance);
}

TF_LITE_MICRO_TEST(CheckCellStateUpdateInt8) {
  const auto gate_output_data = tflite::testing::Get2X2GateOutputCheckData();
  const auto quantization_settings =
      tflite::testing::Get2X2Int8LstmQuantizationSettings();

  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      gate_output_data.input_data, gate_output_data.hidden_state,
      gate_output_data.cell_state);
  auto int8_model_contents = tflite::testing::CreateInt8ModelContents(
      quantization_settings, float_model_contents);

  // get step information: only one time step, no need to update
  // set time_major = true to test batch inference
  auto size_info = tflite::testing::CreateLstmSizeInfo(
      /*time_major*/ true,
      int8_model_contents.GetInternalTensor(tflite::kLstmInputTensor)->dims,
      int8_model_contents.HiddenStateTensor()->dims);
  tflite::lstm_internal::LstmStepManager step_info(size_info);

  // Very high precision. The error is introduced by the
  // quantization error of the clip value (~1e-5), but cannot actually reach
  // the precision due to integer overflow of the elements
  const float tolerance = 1e-3f;
  tflite::testing::TestCellUpdateQuantized<int16_t, 2, 2, 2>(
      step_info, int8_model_contents.CellStateTensor(), gate_output_data,
      quantization_settings, tflite::testing::kModelSettings.cell_clip,
      tolerance);
}

TF_LITE_MICRO_TEST(CheckHiddenStateUpdateInt8) {
  const auto gate_output_data = tflite::testing::Get2X2GateOutputCheckData();
  const auto quantization_settings =
      tflite::testing::Get2X2Int8LstmQuantizationSettings();

  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      gate_output_data.input_data, gate_output_data.hidden_state,
      gate_output_data.expected_updated_cell);
  auto int8_model_contents = tflite::testing::CreateInt8ModelContents(
      quantization_settings, float_model_contents);

  // get step information: only one time step, no need to update
  // set time_major = true to test batch inference
  auto size_info = tflite::testing::CreateLstmSizeInfo(
      /*time_major*/ true,
      int8_model_contents.GetInternalTensor(tflite::kLstmInputTensor)->dims,
      int8_model_contents.HiddenStateTensor()->dims);
  tflite::lstm_internal::LstmStepManager step_info(size_info);

  // Theoritical error floor = quantization scale = 0.004705882165580988
  const float tolerance = 1e-2;

  tflite::testing::TestHiddenStateUpdateQuantized<int8_t, int16_t, 2, 2, 2>(
      step_info, int8_model_contents.CellStateTensor(),
      int8_model_contents.HiddenStateTensor(), gate_output_data,
      quantization_settings, tolerance);
}

TF_LITE_MICRO_TEST(CheckOneStepLSTMInt8) {
  const auto gate_output_data = tflite::testing::Get2X2GateOutputCheckData();
  const auto quantization_settings =
      tflite::testing::Get2X2Int8LstmQuantizationSettings();

  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      gate_output_data.input_data, gate_output_data.hidden_state,
      gate_output_data.cell_state);
  auto int8_model_contents = tflite::testing::CreateInt8ModelContents(
      quantization_settings, float_model_contents);

  const float hidden_state_tolerance = 1e-2;
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-1;
  tflite::testing::TestOneStepLSTMInteger<int8_t, int32_t, int16_t, 2, 3, 2, 2>(
      tflite::testing::kModelSettings, quantization_settings, gate_output_data,
      hidden_state_tolerance, cell_state_tolerance, int8_model_contents);
}

TF_LITE_MICRO_TEST(TestLSTMEvalInt8) {
  const auto kernel_eval_data = tflite::testing::Get2X2LstmEvalCheckData();
  const auto quantization_settings =
      tflite::testing::Get2X2Int8LstmQuantizationSettings();

  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      kernel_eval_data.input_data, kernel_eval_data.hidden_state);
  auto int8_model_contents = tflite::testing::CreateInt8ModelContents(
      quantization_settings, float_model_contents);

  const float hidden_state_tolerance = 1e-2;
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-1;
  tflite::testing::TestLSTMEvalQuantized(
      tflite::testing::kModelSettings, quantization_settings, kernel_eval_data,
      hidden_state_tolerance, cell_state_tolerance, int8_model_contents);
}

#endif  // !defined(XTENSA)
TF_LITE_MICRO_TESTS_END
