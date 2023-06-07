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

#include "signal/micro/kernels/window_flexbuffers_generated_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

namespace {

TfLiteStatus TestWindow(int* input1_dims_data, const int16_t* input1_data,
                        int* input2_dims_data, const int16_t* input2_data,
                        int* output_dims_data, const int16_t* golden,
                        const unsigned char* flexbuffers_data,
                        const unsigned int flexbuffers_data_size,
                        int16_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int kInputsSize = 1;
  constexpr int kOutputsSize = 2;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const int output_len = ElementCount(*output_dims);

  TFLMRegistration* registration = tflite::tflm_signal::Register_WINDOW();
  micro::KernelRunner runner(*registration, tensors, kTensorsSize, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_ENSURE_STATUS(runner.InitAndPrepare(
      reinterpret_cast<const char*>(flexbuffers_data), flexbuffers_data_size));

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

TF_LITE_MICRO_TEST(WindowTestLength16Shift12) {
  int input1_shape[] = {1, 16};
  int input2_shape[] = {1, 16};
  int output_shape[] = {1, 16};
  const int16_t input1[] = {0x000, 0x100, 0x200, 0x300, 0x400, 0x500,
                            0x600, 0x700, 0x800, 0x900, 0xA00, 0xB00,
                            0xC00, 0xD00, 0xE00, 0xF00};
  const int16_t input2[] = {0xF00, 0xE00, 0xD00, 0xC00, 0xB00, 0xA00,
                            0x900, 0x800, 0x700, 0x600, 0x500, 0x400,
                            0x300, 0x200, 0x100, 0x000};
  const int16_t golden[] = {0,   224, 416, 576, 704, 800, 864, 896,
                            896, 864, 800, 704, 576, 416, 224, 0};

  int16_t output[16];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestWindow(
          input1_shape, input1, input2_shape, input2, output_shape, golden,
          g_gen_data_window_shift_12, g_gen_data_size_window_shift_12, output));
}

TF_LITE_MICRO_TEST(WindowTestLength32Shift8) {
  int input1_shape[] = {1, 32};
  int input2_shape[] = {1, 32};
  int output_shape[] = {1, 32};
  const int16_t input1[] = {1221, 1920, 9531, 2795, 1826, 371,  8446, 850,
                            3129, 8218, 4199, 8358, 205,  5268, 3263, 2849,
                            8398, 1381, 6305, 668,  8867, 4651, 9121, 6141,
                            1961, 3750, 8418, 8085, 3308, 1788, 1608, 4761};
  const int16_t input2[] = {1323, 764,  9100, 4220, 1745, 9311, 178,  9442,
                            5676, 1817, 5433, 5837, 7635, 4539, 6548, 9690,
                            6097, 4275, 1523, 3694, 7506, 2797, 5153, 172,
                            2172, 4540, 6643, 7845, 1719, 7564, 1700, 5227};
  const int16_t golden[] = {6310,  5730,  32767, 32767, 12446, 13493, 5872,
                            31350, 32767, 32767, 32767, 32767, 6113,  32767,
                            32767, 32767, 32767, 23061, 32767, 9639,  32767,
                            32767, 32767, 4125,  16637, 32767, 32767, 32767,
                            22212, 32767, 10678, 32767};

  int16_t output[32];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestWindow(
          input1_shape, input1, input2_shape, input2, output_shape, golden,
          g_gen_data_window_shift_8, g_gen_data_size_window_shift_8, output));
}

TF_LITE_MICRO_TEST(WindowTestLength16Shift12OuterDims4) {
  const int kOuterDim = 2;
  int input1_shape[] = {3, kOuterDim, kOuterDim, 16};
  int input2_shape[] = {1, 16};
  int output_shape[] = {3, kOuterDim, kOuterDim, 16};
  const int16_t input1[] = {
      0x000, 0x100, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x800, 0x900,
      0xA00, 0xB00, 0xC00, 0xD00, 0xE00, 0xF00, 0x000, 0x100, 0x200, 0x300,
      0x400, 0x500, 0x600, 0x700, 0x800, 0x900, 0xA00, 0xB00, 0xC00, 0xD00,
      0xE00, 0xF00, 0x000, 0x100, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700,
      0x800, 0x900, 0xA00, 0xB00, 0xC00, 0xD00, 0xE00, 0xF00, 0x000, 0x100,
      0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x800, 0x900, 0xA00, 0xB00,
      0xC00, 0xD00, 0xE00, 0xF00};
  const int16_t input2[] = {0xF00, 0xE00, 0xD00, 0xC00, 0xB00, 0xA00,
                            0x900, 0x800, 0x700, 0x600, 0x500, 0x400,
                            0x300, 0x200, 0x100, 0x000};
  const int16_t golden[] = {
      0,   224, 416, 576, 704, 800, 864, 896, 896, 864, 800, 704, 576,
      416, 224, 0,   0,   224, 416, 576, 704, 800, 864, 896, 896, 864,
      800, 704, 576, 416, 224, 0,   0,   224, 416, 576, 704, 800, 864,
      896, 896, 864, 800, 704, 576, 416, 224, 0,   0,   224, 416, 576,
      704, 800, 864, 896, 896, 864, 800, 704, 576, 416, 224, 0};

  int16_t output[kOuterDim * kOuterDim * 16];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestWindow(
          input1_shape, input1, input2_shape, input2, output_shape, golden,
          g_gen_data_window_shift_12, g_gen_data_size_window_shift_12, output));
}

TF_LITE_MICRO_TESTS_END
