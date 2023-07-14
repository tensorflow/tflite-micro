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

#include "signal/micro/kernels/filter_bank_spectral_subtraction_flexbuffers_generated_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kInputsSize = 1;
constexpr int kOutputsSize = 2;
constexpr int kTensorsSize = kInputsSize + kOutputsSize;

// Speicalized Kernel Runner for running test for the Filter Bank Spectral
// Subtract
//  OP .
class FilterBankSpectralSubtractKernelRunner {
 public:
  explicit FilterBankSpectralSubtractKernelRunner(int* input_dims_data,
                                                  const uint32_t* input_data,
                                                  int* output_dims_data,
                                                  uint32_t* output_data1,
                                                  uint32_t* output_data2)
      : inputs_array_(IntArrayFromInts(inputs_array_data_)),
        outputs_array_(IntArrayFromInts(outputs_array_data_)),
        kernel_runner_(*registration_, tensors_, kTensorsSize, inputs_array_,
                       outputs_array_, nullptr) {
    tensors_[0] = tflite::testing::CreateTensor(
        input_data, tflite::testing::IntArrayFromInts(input_dims_data));

    tensors_[1] = tflite::testing::CreateTensor(
        output_data1, tflite::testing::IntArrayFromInts(output_dims_data));

    tensors_[2] = tflite::testing::CreateTensor(
        output_data2, tflite::testing::IntArrayFromInts(output_dims_data));
  }

  tflite::micro::KernelRunner& kernel_runner() { return kernel_runner_; }

 private:
  int inputs_array_data_[kInputsSize + 1] = {1, 0};
  int outputs_array_data_[kOutputsSize + 1] = {2, 1, 2};
  TfLiteTensor tensors_[kTensorsSize] = {};
  TfLiteIntArray* inputs_array_ = nullptr;
  TfLiteIntArray* outputs_array_ = nullptr;
  TFLMRegistration* registration_ =
      tflite::tflm_signal::Register_FILTER_BANK_SPECTRAL_SUBTRACTION();
  tflite::micro::KernelRunner kernel_runner_;
};

TfLiteStatus TestFilterBankSpectralSubtractionInvoke(
    int* output_dims_data, const uint32_t* golden1, const uint32_t* golden2,
    uint32_t* output1_data, uint32_t* output2_data,
    tflite::micro::KernelRunner& kernel_runner) {
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_len = ElementCount(*output_dims);

  TF_LITE_ENSURE_STATUS(kernel_runner.Invoke());

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden1[i], output1_data[i]);
    TF_LITE_MICRO_EXPECT_EQ(golden2[i], output2_data[i]);
  }

  return kTfLiteOk;
}

TfLiteStatus TestFilterBankSpectralSubtraction(
    int* input_dims_data, const uint32_t* input_data, int* output_dims_data,
    const uint32_t* golden1, const uint32_t* golden2,
    const uint8_t* flexbuffers_data, const int flexbuffers_data_len,
    uint32_t* output1_data, uint32_t* output2_data) {
  FilterBankSpectralSubtractKernelRunner filter_bank_spectral_subtract_runner(
      input_dims_data, input_data, output_dims_data, output1_data,
      output2_data);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_ENSURE_STATUS(
      filter_bank_spectral_subtract_runner.kernel_runner().InitAndPrepare(
          reinterpret_cast<const char*>(flexbuffers_data),
          flexbuffers_data_len));

  TF_LITE_ENSURE_STATUS(TestFilterBankSpectralSubtractionInvoke(
      output_dims_data, golden1, golden2, output1_data, output2_data,
      filter_bank_spectral_subtract_runner.kernel_runner()));

  return kTfLiteOk;
}

TfLiteStatus TestFilterBankSpectralSubtractionReset(
    int* input_dims_data, const uint32_t* input_data, int* output_dims_data,
    const uint32_t* golden1, const uint32_t* golden2,
    const uint8_t* flexbuffers_data, const int flexbuffers_data_len,
    uint32_t* output1_data, uint32_t* output2_data) {
  FilterBankSpectralSubtractKernelRunner filter_bank_spectral_subtract_runner(
      input_dims_data, input_data, output_dims_data, output1_data,
      output2_data);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_ENSURE_STATUS(
      filter_bank_spectral_subtract_runner.kernel_runner().InitAndPrepare(
          reinterpret_cast<const char*>(flexbuffers_data),
          flexbuffers_data_len));

  TF_LITE_ENSURE_STATUS(TestFilterBankSpectralSubtractionInvoke(
      output_dims_data, golden1, golden2, output1_data, output2_data,
      filter_bank_spectral_subtract_runner.kernel_runner()));
  filter_bank_spectral_subtract_runner.kernel_runner().Reset();
  TF_LITE_ENSURE_STATUS(TestFilterBankSpectralSubtractionInvoke(
      output_dims_data, golden1, golden2, output1_data, output2_data,
      filter_bank_spectral_subtract_runner.kernel_runner()));

  return kTfLiteOk;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FilterBankSpectralSubtractionTest32Channel) {
  int input_shape[] = {1, 32};
  int output_shape[] = {1, 32};

  const uint32_t input[] = {322, 308, 210, 212, 181, 251, 403, 259, 65,  48, 76,
                            48,  50,  46,  53,  52,  112, 191, 136, 59,  70, 51,
                            39,  64,  33,  44,  41,  49,  74,  107, 262, 479};
  const uint32_t golden1[] = {310, 296, 202, 204, 174, 241, 387, 249,
                              63,  47,  73,  47,  49,  45,  51,  50,
                              108, 184, 131, 57,  68,  49,  38,  62,
                              32,  43,  40,  48,  72,  103, 252, 460};
  const uint32_t golden2[] = {12, 12, 8, 8, 7, 10, 16, 10, 2,  1, 3,
                              1,  1,  1, 2, 2, 4,  7,  5,  2,  2, 2,
                              1,  2,  1, 1, 1, 1,  2,  4,  10, 19};
  uint32_t output1[32];
  uint32_t output2[32];
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFilterBankSpectralSubtraction(
          input_shape, input, output_shape, golden1, golden2,
          g_gen_data_filter_bank_spectral_subtraction_32_channel,
          g_gen_data_size_filter_bank_spectral_subtraction_32_channel, output1,
          output2));
}

TF_LITE_MICRO_TEST(FilterBankSpectralSubtractionTest16Channel) {
  int input_shape[] = {1, 16};
  int output_shape[] = {1, 16};

  const uint32_t input[] = {393, 213, 408, 1,   361, 385, 386, 326,
                            170, 368, 368, 305, 152, 322, 213, 319};
  const uint32_t golden1[] = {378, 205, 392, 1,   347, 370, 371, 313,
                              164, 354, 354, 293, 146, 310, 205, 307};
  const uint32_t golden2[] = {15, 8,  16, 0,  14, 15, 15, 13,
                              6,  14, 14, 12, 6,  12, 8,  12};
  uint32_t output1[32];
  uint32_t output2[32];
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFilterBankSpectralSubtraction(
          input_shape, input, output_shape, golden1, golden2,
          g_gen_data_filter_bank_spectral_subtraction_16_channel,
          g_gen_data_size_filter_bank_spectral_subtraction_16_channel, output1,
          output2));
}

TF_LITE_MICRO_TEST(FilterBankSpectralSubtractionTest32ChannelReset) {
  int input_shape[] = {1, 32};
  int output_shape[] = {1, 32};

  const uint32_t input[] = {322, 308, 210, 212, 181, 251, 403, 259, 65,  48, 76,
                            48,  50,  46,  53,  52,  112, 191, 136, 59,  70, 51,
                            39,  64,  33,  44,  41,  49,  74,  107, 262, 479};
  const uint32_t golden1[] = {310, 296, 202, 204, 174, 241, 387, 249,
                              63,  47,  73,  47,  49,  45,  51,  50,
                              108, 184, 131, 57,  68,  49,  38,  62,
                              32,  43,  40,  48,  72,  103, 252, 460};
  const uint32_t golden2[] = {12, 12, 8, 8, 7, 10, 16, 10, 2,  1, 3,
                              1,  1,  1, 2, 2, 4,  7,  5,  2,  2, 2,
                              1,  2,  1, 1, 1, 1,  2,  4,  10, 19};

  uint32_t output1[32];
  uint32_t output2[32];
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFilterBankSpectralSubtractionReset(
          input_shape, input, output_shape, golden1, golden2,
          g_gen_data_filter_bank_spectral_subtraction_32_channel,
          g_gen_data_size_filter_bank_spectral_subtraction_32_channel, output1,
          output2));
}

TF_LITE_MICRO_TEST(FilterBankSpectralSubtractionTest16ChannelReset) {
  int input_shape[] = {1, 16};
  int output_shape[] = {1, 16};

  const uint32_t input[] = {393, 213, 408, 1,   361, 385, 386, 326,
                            170, 368, 368, 305, 152, 322, 213, 319};
  const uint32_t golden1[] = {378, 205, 392, 1,   347, 370, 371, 313,
                              164, 354, 354, 293, 146, 310, 205, 307};
  const uint32_t golden2[] = {15, 8,  16, 0,  14, 15, 15, 13,
                              6,  14, 14, 12, 6,  12, 8,  12};
  uint32_t output1[32];
  uint32_t output2[32];
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFilterBankSpectralSubtractionReset(
          input_shape, input, output_shape, golden1, golden2,
          g_gen_data_filter_bank_spectral_subtraction_16_channel,
          g_gen_data_size_filter_bank_spectral_subtraction_16_channel, output1,
          output2));
}

TF_LITE_MICRO_TESTS_END
