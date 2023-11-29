/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdio>

#include "signal/micro/kernels/fft_flexbuffers_generated_data.h"
#include "signal/testdata/fft_test_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

namespace {

template <typename T>
TfLiteStatus ValidateFFTGoldens(
    TfLiteTensor* tensors, const int tensors_size, TfLiteIntArray* inputs_array,
    TfLiteIntArray* outputs_array, int output_len, const T* golden,
    const TFLMRegistration registration, const uint8_t* flexbuffers_data,
    const int flexbuffers_data_len, T* output_data, T tolerance) {
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);
  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TfLiteStatus status = runner.InitAndPrepare(
      reinterpret_cast<const char*>(flexbuffers_data), flexbuffers_data_len);
  if (status != kTfLiteOk) {
    return status;
  }
  status = runner.Invoke();
  if (status != kTfLiteOk) {
    return status;
  }
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus TestFFT(int* input_dims_data, const T* input_data,
                     int* output_dims_data, const T* golden,
                     const TFLMRegistration registration,
                     const uint8_t* flexbuffers_data,
                     const int flexbuffers_data_len, T* output_data,
                     T tolerance) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

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
  const int output_len = ElementCount(*output_dims);

  TF_LITE_ENSURE_STATUS(
      ValidateFFTGoldens<T>(tensors, kTensorsSize, inputs_array, outputs_array,
                            output_len, golden, registration, flexbuffers_data,
                            flexbuffers_data_len, output_data, tolerance));

  return kTfLiteOk;
}

TfLiteStatus TestFFTAutoScale(int* input_dims_data, const int16_t* input_data,
                              int* output_dims_data, const int16_t* golden,
                              int* scale_bit_dims_data,
                              const int32_t scale_bit_golden,
                              const TFLMRegistration registration,
                              const uint8_t* flexbuffers_data,
                              const int flexbuffers_data_len,
                              int16_t* output_data, int32_t* scale_bit) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  TfLiteIntArray* scale_bit_dims = IntArrayFromInts(scale_bit_dims_data);

  constexpr int kInputsSize = 1;
  constexpr int kOutputsSize = 2;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
      CreateTensor(scale_bit, scale_bit_dims),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {2, 1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const int output_len = ElementCount(*output_dims);

  TF_LITE_ENSURE_STATUS(ValidateFFTGoldens<int16_t>(
      tensors, kTensorsSize, inputs_array, outputs_array, output_len, golden,
      registration, flexbuffers_data, flexbuffers_data_len, output_data, 0));

  TF_LITE_MICRO_EXPECT_EQ(scale_bit_golden, *scale_bit);

  return kTfLiteOk;
}

}  // namespace

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(RfftTestSize64Float) {
  constexpr int kOutputLen = 66;
  int input_shape[] = {1, 64};
  const float input[] = {16384., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int output_shape[] = {1, kOutputLen};
  const float golden[] = {16384., 0, 16384., 0, 16384., 0, 16384., 0, 16384., 0,
                          16384., 0, 16384., 0, 16384., 0, 16384., 0, 16384., 0,
                          16384., 0, 16384., 0, 16384., 0, 16384., 0, 16384., 0,
                          16384., 0, 16384., 0, 16384., 0, 16384., 0, 16384., 0,
                          16384., 0, 16384., 0, 16384., 0, 16384., 0, 16384., 0,
                          16384., 0, 16384., 0, 16384., 0, 16384., 0, 16384., 0,
                          16384., 0, 16384., 0, 16384., 0};
  float output[kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_RFFT_FLOAT();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<float>(
                     input_shape, input, output_shape, golden, *registration,
                     g_gen_data_fft_length_64_float,
                     g_gen_data_size_fft_length_64_float, output, 1e-7));
}

#if !defined __XTENSA__
// Currently, only 16-bit RFFT of size 512 is supported.
TF_LITE_MICRO_TEST(RfftTestSize64Int16) {
  constexpr int kOutputLen = 66;
  int input_shape[] = {1, 64};
  const int16_t input[] = {16384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int output_shape[] = {1, kOutputLen};
  const int16_t golden[] = {
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0};
  int16_t output[kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_RFFT_INT16();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<int16_t>(
                     input_shape, input, output_shape, golden, *registration,
                     g_gen_data_fft_length_64_int16,
                     g_gen_data_size_fft_length_64_int16, output, 0));
}
#endif

TF_LITE_MICRO_TEST(RfftTestSize64Int32) {
  constexpr int kOutputLen = 66;
  int input_shape[] = {1, 64};
  const int32_t input[] = {16384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int output_shape[] = {1, kOutputLen};
  const int32_t golden[] = {
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0};
  int32_t output[kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_RFFT_INT32();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<int32_t>(
                     input_shape, input, output_shape, golden, *registration,
                     g_gen_data_fft_length_64_int32,
                     g_gen_data_size_fft_length_64_int32, output, 0));
}

TF_LITE_MICRO_TEST(RfftTestSize64Int32OuterDims4) {
  constexpr int kOutputLen = 66;
  constexpr int kOuterDim = 2;
  int input_shape[] = {3, kOuterDim, kOuterDim, 64};
  const int32_t input[] = {
      16384, 0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 16384, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 16384, 0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 16384, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0,
      0,     0, 0,     0, 0, 0, 0, 0,     0};
  int output_shape[] = {3, kOuterDim, kOuterDim, kOutputLen};
  const int32_t golden[] = {
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0};
  int32_t output[kOuterDim * kOuterDim * kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_RFFT_INT32();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<int32_t>(
                     input_shape, input, output_shape, golden, *registration,
                     g_gen_data_fft_length_64_int32,
                     g_gen_data_size_fft_length_64_int32, output, 0));
}

TF_LITE_MICRO_TEST(RfftTestSize512Float) {
  constexpr int kOutputLen = 514;
  int input_shape[] = {1, 512};
  int output_shape[] = {1, kOutputLen};
  // Outputs are ComplexInt16 which takes twice the space as regular int16.
  float output[kOutputLen * 2];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_RFFT_FLOAT();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<float>(
                     input_shape, tflite::kRfftFloatLength512Input,
                     output_shape, tflite::kRfftFloatLength512Golden,
                     *registration, g_gen_data_fft_length_512_float,
                     g_gen_data_size_fft_length_512_float, output, 1e-5));
}

TF_LITE_MICRO_TEST(RfftTestSize512Int16) {
  constexpr int kOutputLen = 514;
  int input_shape[] = {1, 512};
  int output_shape[] = {1, kOutputLen};
  // Outputs are ComplexInt16 which takes twice the space as regular int16.
  int16_t output[kOutputLen * 2];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_RFFT_INT16();
// See (b/287518815) for why this is needed.
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  int tolerance = 9;
#else   // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  int tolerance = 3;
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<int16_t>(
                     input_shape, tflite::kRfftInt16Length512Input,
                     output_shape, tflite::kRfftInt16Length512Golden,
                     *registration, g_gen_data_fft_length_512_int16,
                     g_gen_data_size_fft_length_512_int16, output, tolerance));
}

TF_LITE_MICRO_TEST(RfftTestSize512Int32) {
  constexpr int kOutputLen = 514;
  int input_shape[] = {1, 512};
  int output_shape[] = {1, kOutputLen};
  // Outputs are ComplexInt32 which takes twice the space as regular int32.
  int32_t output[kOutputLen * 2];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_RFFT_INT32();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::TestFFT<int32_t>(
                              input_shape, tflite::kRfftInt32Length512Input,
                              output_shape, tflite::kRfftInt32Length512Golden,
                              *registration, g_gen_data_fft_length_512_int32,
                              g_gen_data_size_fft_length_512_int32, output, 0));
}

TF_LITE_MICRO_TEST(IrfftTestLength64Float) {
  constexpr int kOutputLen = 64;
  int input_shape[] = {1, 66};
  const float input[] = {256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                         256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                         256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                         256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
                         256, 0, 256, 0, 256, 0, 256, 0, 256, 0};
  int output_shape[] = {1, kOutputLen};
  const float golden[] = {256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float output[kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_IRFFT_FLOAT();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<float>(
                     input_shape, input, output_shape, golden, *registration,
                     g_gen_data_fft_length_64_float,
                     g_gen_data_size_fft_length_64_int16, output, 1e-7));
}

TF_LITE_MICRO_TEST(IrfftTestLength64Int16) {
  constexpr int kOutputLen = 64;
  int input_shape[] = {1, 66};
  const int16_t input[] = {
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0};
  int output_shape[] = {1, kOutputLen};
  const int16_t golden[] = {256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int16_t output[kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_IRFFT_INT16();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<int16_t>(
                     input_shape, input, output_shape, golden, *registration,
                     g_gen_data_fft_length_64_int16,
                     g_gen_data_size_fft_length_64_int16, output, 0));
}

TF_LITE_MICRO_TEST(IrfftTestLength64Int32) {
  constexpr int kOutputLen = 64;
  int input_shape[] = {1, 66};
  const int32_t input[] = {
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0};
  int output_shape[] = {1, kOutputLen};
  const int32_t golden[] = {256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int32_t output[kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_IRFFT_INT32();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<int32_t>(
                     input_shape, input, output_shape, golden, *registration,
                     g_gen_data_fft_length_64_int32,
                     g_gen_data_size_fft_length_64_int32, output, 0));
}

TF_LITE_MICRO_TEST(IrfftTestLength64Int32OuterDims4) {
  constexpr int kOutputLen = 64;
  constexpr int kOuterDim = 2;
  int input_shape[] = {3, kOuterDim, kOuterDim, 66};
  const int32_t input[] = {
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0,
      256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0};
  int output_shape[] = {3, kOuterDim, kOuterDim, kOutputLen};
  const int32_t golden[] = {
      256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0,   0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0,   0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 256, 0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0,   0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0,   0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 256, 0, 0,   0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0,   0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0,   0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 256, 0, 0,   0, 0,   0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0,   0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0,   0,
      0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int32_t output[kOuterDim * kOuterDim * kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_IRFFT_INT32();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<int32_t>(
                     input_shape, input, output_shape, golden, *registration,
                     g_gen_data_fft_length_64_int32,
                     g_gen_data_size_fft_length_64_int32, output, 0));
}

TF_LITE_MICRO_TEST(IrfftTestLength512Float) {
  constexpr int kOutputLen = 512;
  int input_shape[] = {1, 514};
  int output_shape[] = {1, kOutputLen};
  float output[kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_IRFFT_FLOAT();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestFFT<float>(
                     input_shape, tflite::kIrfftFloatLength512Input,
                     output_shape, tflite::kIrfftFloatLength512Golden,
                     *registration, g_gen_data_fft_length_512_float,
                     g_gen_data_size_fft_length_512_float, output, 1e-6));
}

TF_LITE_MICRO_TEST(IrfftTestLength512Int16) {
  constexpr int kOutputLen = 512;
  int input_shape[] = {1, 514};
  int output_shape[] = {1, kOutputLen};
  int16_t output[kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_IRFFT_INT16();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::TestFFT<int16_t>(
                              input_shape, tflite::kIrfftInt16Length512Input,
                              output_shape, tflite::kIrfftInt16Length512Golden,
                              *registration, g_gen_data_fft_length_512_int16,
                              g_gen_data_size_fft_length_512_int16, output, 0));
}

TF_LITE_MICRO_TEST(IrfftTestLength512Int32) {
  constexpr int kOutputLen = 512;
  int input_shape[] = {1, 514};
  int output_shape[] = {1, kOutputLen};
  int32_t output[kOutputLen];
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_IRFFT_INT32();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::TestFFT<int32_t>(
                              input_shape, tflite::kIrfftInt32Length512Input,
                              output_shape, tflite::kIrfftInt32Length512Golden,
                              *registration, g_gen_data_fft_length_512_int32,
                              g_gen_data_size_fft_length_512_int32, output, 0));
}

TF_LITE_MICRO_TEST(FftAutoScaleTestSmall) {
  constexpr int kTensorsSize = 8;
  int shape[] = {1, 8};
  const int16_t input[] = {0x0000, 0x1111, 0x2222, 0x3333,
                           0x3333, 0x2222, 0x1111, 0x0000};
  int16_t output[kTensorsSize];
  int scale_bit_shape[] = {0};
  int32_t scale_bit;
  const int16_t golden[] = {0x0000, 0x2222, 0x4444, 0x6666,
                            0x6666, 0x4444, 0x2222, 0x0000};
  const int32_t scale_bit_golden = 1;
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_FFT_AUTO_SCALE();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFFTAutoScale(
          shape, input, shape, golden, scale_bit_shape, scale_bit_golden,
          *registration, nullptr, 0, output, &scale_bit));
}

TF_LITE_MICRO_TEST(FftAutoScaleTestScaleBit) {
  constexpr int kTensorsSize = 8;
  int shape[] = {1, 8};
  const int16_t input[] = {238, 113, -88, -243, -5, -130, 159, -70};
  int16_t output[kTensorsSize];
  int scale_bit_shape[] = {0};
  int32_t scale_bit;
  const int16_t golden[] = {30464, 14464,  -11264, -31104,
                            -640,  -16640, 20352,  -8960};
  const int32_t scale_bit_golden = 7;
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_FFT_AUTO_SCALE();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFFTAutoScale(
          shape, input, shape, golden, scale_bit_shape, scale_bit_golden,
          *registration, nullptr, 0, output, &scale_bit));
}

TF_LITE_MICRO_TEST(FftAutoScaleTestLarge) {
  constexpr int kTensorsSize = 400;
  int shape[] = {1, kTensorsSize};
  int16_t output[kTensorsSize];
  int scale_bit_shape[] = {0};
  int32_t scale_bit;

  const int32_t scale_bit_golden = 0;
  const TFLMRegistration* registration =
      tflite::tflm_signal::Register_FFT_AUTO_SCALE();
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestFFTAutoScale(
          shape, tflite::kFftAutoScaleLength512Input, shape,
          tflite::kFftAutoScaleLength512Golden, scale_bit_shape,
          scale_bit_golden, *registration, nullptr, 0, output, &scale_bit));
}

TF_LITE_MICRO_TESTS_END
