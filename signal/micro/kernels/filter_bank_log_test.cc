/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include "signal/micro/kernels/filter_bank_log_flexbuffers_generated_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

TfLiteStatus TestFilterBankLog(int* input_dims_data, const uint32_t* input_data,
                               int* output_dims_data, const int16_t* golden,
                               const uint8_t* flexbuffers_data,
                               const int flexbuffers_data_len,
                               int16_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int kInputsSize = 1;
  constexpr int kOutputsSize = 2;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const int output_len = ElementCount(*output_dims);

  TFLMRegistration* registration =
      tflite::tflm_signal::Register_FILTER_BANK_LOG();
  micro::KernelRunner runner(*registration, tensors, kTensorsSize, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_ENSURE_STATUS(runner.InitAndPrepare(
      reinterpret_cast<const char*>(flexbuffers_data), flexbuffers_data_len));

  TF_LITE_ENSURE_STATUS(runner.Invoke());

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output_data[i]);
  }

  return kTfLiteOk;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FilterBankLogTest32Channel) {
  int input_shape[] = {1, 32};
  int output_shape[] = {1, 32};
  const uint32_t input[] = {29, 21, 29, 40, 19, 11, 13, 23, 13, 11, 25,
                            17, 5,  4,  46, 14, 17, 14, 20, 14, 10, 10,
                            15, 11, 17, 12, 15, 16, 19, 18, 6,  2};
  const int16_t golden[] = {8715, 8198, 8715, 9229, 8038, 7164, 7431, 8344,
                            7431, 7164, 8477, 7860, 5902, 5545, 9453, 7550,
                            7860, 7550, 8120, 7550, 7011, 7011, 7660, 7164,
                            7860, 7303, 7660, 7763, 8038, 7952, 6194, 4436};
  int16_t output[32];
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFilterBankLog(
          input_shape, input, output_shape, golden,
          g_gen_data_filter_bank_log_scale_1600_correction_bits_3,
          g_gen_data_size_filter_bank_log_scale_1600_correction_bits_3,
          output));
}

TF_LITE_MICRO_TEST(FilterBankLogTest16Channel) {
  int input_shape[] = {1, 16};
  int output_shape[] = {1, 16};
  const uint32_t input[] = {48, 20, 19, 24, 35, 47, 23, 30,
                            31, 10, 48, 21, 46, 14, 18, 27};
  const int16_t golden[] = {32767, 15121, 13440, 21095, 32767, 32767,
                            19701, 28407, 29482, 32767, 32767, 16720,
                            32767, 3434,  11669, 24955};
  int16_t output[16];
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFilterBankLog(
          input_shape, input, output_shape, golden,
          g_gen_data_filter_bank_log_scale_32768_correction_bits_5,
          g_gen_data_size_filter_bank_log_scale_32768_correction_bits_5,
          output));
}

TF_LITE_MICRO_TESTS_END
