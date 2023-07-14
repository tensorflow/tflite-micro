/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "signal/micro/kernels/energy_flexbuffers_generated_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

TfLiteStatus TestEnergy(int* input_dims_data, const int16_t* input_data,
                        int* output_dims_data, const uint32_t* golden,
                        const unsigned char* flexbuffers_data,
                        const unsigned int flexbuffers_data_size,
                        uint32_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_len = ElementCount(*output_dims);
  constexpr int kInputsSize = 1;
  constexpr int kOutputsSize = 1;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration* registration = tflm_signal::Register_ENERGY();
  micro::KernelRunner runner(*registration, tensors, kTensorsSize, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TfLiteStatus status = runner.InitAndPrepare(
      reinterpret_cast<const char*>(flexbuffers_data), flexbuffers_data_size);
  if (status != kTfLiteOk) {
    return status;
  }

  status = runner.Invoke();
  if (status != kTfLiteOk) {
    return status;
  }

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output_data[i]);
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(EnergyTestMiddle) {
  int input_shape[] = {1, 16};
  int output_shape[] = {1, 8};
  const int16_t input[] = {1, 2,  3,  4,  5,  6,  7,  8,
                           9, 10, 11, 12, 13, 14, 15, 16};
  const uint32_t golden[] = {0, 0, 61, 113, 0, 0, 0, 0};
  uint32_t output[8];
  memset(output, 0, sizeof(output));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestEnergy(
                     input_shape, input, output_shape, golden,
                     g_gen_data_start_index_2_end_index_4,
                     g_gen_data_size_start_index_2_end_index_4, output));
}

TF_LITE_MICRO_TEST(EnergyTestStart) {
  int input_shape[] = {1, 16};
  int output_shape[] = {1, 8};
  const int16_t input[] = {1, 2,  3,  4,  5,  6,  7,  8,
                           9, 10, 11, 12, 13, 14, 15, 16};
  const uint32_t golden[] = {5, 25, 61, 113, 0, 0, 0, 0};
  uint32_t output[8];
  memset(output, 0, sizeof(output));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestEnergy(
                     input_shape, input, output_shape, golden,
                     g_gen_data_start_index_0_end_index_4,
                     g_gen_data_size_start_index_0_end_index_4, output));
}

TF_LITE_MICRO_TEST(EnergyTestEnd) {
  int input_shape[] = {1, 16};
  int output_shape[] = {1, 8};
  const int16_t input[] = {1, 2,  3,  4,  5,  6,  7,  8,
                           9, 10, 11, 12, 13, 14, 15, 16};
  const uint32_t golden[] = {0, 0, 0, 0, 181, 265, 365, 481};
  uint32_t output[8];
  memset(output, 0, sizeof(output));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestEnergy(
                     input_shape, input, output_shape, golden,
                     g_gen_data_start_index_4_end_index_8,
                     g_gen_data_size_start_index_4_end_index_8, output));
}

TF_LITE_MICRO_TESTS_END
