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

#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

TfLiteStatus TestFilterBankSquareRoot(
    int* input1_dims_data, const uint64_t* input1_data, int* input2_dims_data,
    const int32_t* input2_data, int* output_dims_data, const uint32_t* golden,
    uint32_t* output_data) {
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

  TFLMRegistration* registration =
      tflite::tflm_signal::Register_FILTER_BANK_SQUARE_ROOT();
  micro::KernelRunner runner(*registration, tensors, kTensorsSize, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_ENSURE_STATUS(runner.InitAndPrepare(nullptr, 0));

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

TF_LITE_MICRO_TEST(FilterBankSquareRoot32Channel) {
  int input1_shape[] = {1, 32};
  int input2_shape[] = {0};
  int output_shape[] = {1, 32};
  const uint64_t input1[] = {
      10528000193, 28362909357, 47577133750, 8466055850, 5842710800, 2350911449,
      2989811430,  2646718839,  515262774,   276394561,  469831522,  55815334,
      28232446,    11591835,    40329249,    67658028,   183446654,  323189165,
      117473797,   41339272,    25846050,    12428673,   18670978,   22521722,
      78477733,    54207503,    25150296,    43098592,   28211625,   15736687,
      20990296,    17907031};
  const int32_t input2[] = {7};
  const uint32_t golden[] = {801, 1315, 1704, 718, 597, 378, 427, 401,
                             177, 129,  169,  58,  41,  26,  49,  64,
                             105, 140,  84,   50,  39,  27,  33,  37,
                             69,  57,   39,   51,  41,  30,  35,  33};
  uint32_t output[32];
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, tflite::testing::TestFilterBankSquareRoot(
                                         input1_shape, input1, input2_shape,
                                         input2, output_shape, golden, output));
}

TF_LITE_MICRO_TEST(FilterBankSquareRoot16Channel) {
  int input1_shape[] = {1, 16};
  int input2_shape[] = {0};
  int output_shape[] = {1, 16};
  const uint64_t input1[] = {
      13051415151, 14932650877, 18954728418, 8730126017,
      6529665275,  12952546517, 10314975609, 8919697835,
      8053663348,  17231208421, 7366899760,  1372112200,
      19953434807, 17012385332, 4710443222,  17765594053};
  const int32_t input2[] = {5};
  const uint32_t golden[] = {3570, 3818, 4302, 2919, 2525, 3556, 3173, 2951,
                             2804, 4102, 2682, 1157, 4414, 4076, 2144, 4165};
  uint32_t output[16];
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, tflite::testing::TestFilterBankSquareRoot(
                                         input1_shape, input1, input2_shape,
                                         input2, output_shape, golden, output));
}

TF_LITE_MICRO_TESTS_END
