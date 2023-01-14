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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
// Validate the output result array with golden values
template <typename T>
void ValidateResultGoldens(const T* golden, const T* output_data,
                           const int output_len, const float tolerance) {
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void TestUnidirectionalLSTM(
    const TfLiteUnidirectionalSequenceLSTMParams& builtin_data,
    const ModelQuantizationParameters& quantization_settings,
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    ModelContents<ActivationType, WeightType, BiasType, CellType, batch_size,
                  time_steps, input_dimension, state_dimension>&
        model_contents) {
  const TfLiteRegistration registration =
      Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INTEGER();
  micro::KernelRunner runner(registration, model_contents.tensors(), 24 + 1,
                             model_contents.inputs_array_data,
                             model_contents.outputs_array_data,
                             reinterpret_cast<void*>(&builtin_data));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  float dequantized_hidden_state[batch_size * state_dimension] = {};
  Dequantize(model_contents.GetHiddenStateData(), batch_size * state_dimension,
             quantization_settings.hidden_quantization_parameters.scale,
             quantization_settings.hidden_quantization_parameters.zero_point,
             dequantized_hidden_state);

  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        dequantized_hidden_state, batch_size * state_dimension,
                        hidden_state_tolerance);

  float dequantized_cell_state[batch_size * state_dimension] = {};
  Dequantize(model_contents.GetCellStateData(), batch_size * state_dimension,
             quantization_settings.cell_quantization_parameters.scale,
             quantization_settings.cell_quantization_parameters.zero_point,
             dequantized_cell_state);
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        dequantized_cell_state, batch_size * state_dimension,
                        cell_state_tolerance);

  float dequantized_output[batch_size * state_dimension * time_steps] = {};
  Dequantize(model_contents.GetOutputData(),
             batch_size * state_dimension * time_steps,
             quantization_settings.output_quantization_parameters.scale,
             quantization_settings.output_quantization_parameters.zero_point,
             dequantized_output);
  ValidateResultGoldens(eval_check_data.expected_output, dequantized_output,
                        batch_size * state_dimension, hidden_state_tolerance);
}
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN
// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)

TF_LITE_MICRO_TEST(TestUnidirectionalLSTMInt16) {
  const auto kernel_eval_data = tflite::testing::Get2X2LstmEvalCheckData();
  const auto quantization_settings =
      tflite::testing::Get2X2Int16LstmQuantizationSettings();

  auto float_model_contents = tflite::testing::Create2x3x2X2FloatModelContents(
      kernel_eval_data.input_data, kernel_eval_data.hidden_state);
  //   auto int16_model_contents =
  //       tflite::testing::CreateIntModelContents<int16_t, int8_t, int64_t,
  //                                               int16_t>(quantization_settings,
  //                                                        float_model_contents);

  const float hidden_state_tolerance = 1e-3;  // actually very close to 1e-4
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-2;
  tflite::testing::TestUnidirectionalLSTM(
      quantization_settings, kernel_eval_data, hidden_state_tolerance,
      cell_state_tolerance, int16_model_contents);
}
#endif  // !defined(XTENSA)
TF_LITE_MICRO_TESTS_END