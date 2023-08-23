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

#include "signal/micro/kernels/pcan_flexbuffers_generated_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace tflm_signal {
namespace {

TfLiteStatus TestPCAN(const unsigned char* init_data, int init_data_size,
                      int* input_dims_data, const uint32_t* input_data,
                      int* noise_estimate_dims_data,
                      const uint32_t* noise_estimate_data,
                      int* gain_lut_dims_data, const int16_t* gain_lut_data,
                      int* output_dims_data, const uint32_t* golden,
                      uint32_t* output_data) {
  TfLiteIntArray* input_dims =
      ::tflite::testing::IntArrayFromInts(input_dims_data);
  TfLiteIntArray* noise_estimate_dims =
      ::tflite::testing::IntArrayFromInts(noise_estimate_dims_data);
  TfLiteIntArray* gain_lut_dims =
      ::tflite::testing::IntArrayFromInts(gain_lut_dims_data);
  TfLiteIntArray* output_dims =
      ::tflite::testing::IntArrayFromInts(output_dims_data);
  const int output_len = ElementCount(*output_dims);
  constexpr int kInputsSize = 3;
  constexpr int kOutputsSize = 1;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      tflite::testing::CreateTensor(input_data, input_dims),
      tflite::testing::CreateTensor(noise_estimate_data, noise_estimate_dims),
      tflite::testing::CreateTensor(gain_lut_data, gain_lut_dims),
      tflite::testing::CreateTensor(output_data, output_dims),
  };
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array =
      ::tflite::testing::IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array =
      ::tflite::testing::IntArrayFromInts(outputs_array_data);

  const TFLMRegistration* registration = tflite::tflm_signal::Register_PCAN();
  micro::KernelRunner runner(*registration, tensors, kTensorsSize, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TfLiteStatus status = runner.InitAndPrepare(
      reinterpret_cast<const char*>(init_data), init_data_size);
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
}  // namespace tflm_signal
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(Mix1Ref1Case) {
  int input_shape[] = {1, 40};
  int noise_estimate_shape[] = {1, 40};
  int gain_lut_shape[] = {1, 125};
  int output_shape[] = {1, 40};
  const uint32_t input[] = {286, 298, 305, 291, 290, 279, 273, 257, 250, 240,
                            240, 233, 234, 230, 221, 205, 183, 159, 156, 188,
                            239, 298, 345, 374, 380, 369, 359, 364, 372, 354,
                            302, 243, 194, 135, 64,  72,  171, 245, 277, 304};
  const uint32_t noise_estimate[] = {
      7310, 18308, 7796, 17878, 7413, 17141, 6978, 15789, 6390, 14745,
      6135, 14314, 5981, 14130, 5649, 12594, 4677, 9768,  3987, 11550,
      6109, 18308, 8819, 22977, 9713, 22670, 9176, 22363, 9509, 21748,
      7719, 14929, 4959, 8294,  1636, 4423,  4371, 15052, 7080, 18677};

  const int16_t gain_lut[] = {
      32636, 32633,  32630, -6,     0,     -21589, 32624, -12,    0,     -21589,
      32612, -23,    -2,    -21589, 32587, -48,    0,     -21589, 32539, -96,
      0,     -21589, 32443, -190,   0,     -21589, 32253, -378,   4,     -21589,
      31879, -739,   18,    -21589, 31158, -1409,  62,    -21589, 29811, -2567,
      202,   -21589, 27446, -4301,  562,   -21589, 23707, -6265,  1230,  -21589,
      18672, -7458,  1952,  -21589, 13166, -7030,  2212,  -21589, 8348,  -5342,
      1868,  -21589, 4874,  -3459,  1282,  -21589, 2697,  -2025,  774,   -21589,
      1446,  -1120,  436,   -21589, 762,   -596,   232,   -21589, 398,   -313,
      122,   -21589, 207,   -164,   64,    -21589, 107,   -85,    34,    -21589,
      56,    -45,    18,    -21589, 29,    -22,    8,     -21589, 15,    -13,
      6,     -21589, 8,     -8,     4,     -21589, 4,     -2,     0,     -21589,
      2,     -3,     2,     -21589, 1,     0,      0,     -21589, 1,     -3,
      2,     -21589, 0,     0,      0};

  uint32_t output[40];
  const uint32_t golden[] = {1301, 836, 1354, 827, 1312, 811, 1263, 779,
                             1192, 753, 1160, 743, 1140, 738, 1096, 698,
                             956,  607, 845,  667, 1157, 836, 1461, 912,
                             1546, 908, 1496, 904, 1527, 895, 1346, 758,
                             999,  548, 378,  344, 908,  761, 1274, 843};
  memset(output, 0, sizeof(output));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::tflm_signal::TestPCAN(
          g_gen_data_snr_shift_6_test, g_gen_data_size_snr_shift_6_test,
          input_shape, input, noise_estimate_shape, noise_estimate,
          gain_lut_shape, gain_lut, output_shape, golden, output));
}

TF_LITE_MICRO_TESTS_END
