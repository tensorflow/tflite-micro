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
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {
using ::tflite::testing::CreateTensor;
using ::tflite::testing::IntArrayFromInts;

// The layout of tensors is fixed.
constexpr int kShape1Index = 0;
constexpr int kShape2Index = 1;
constexpr int kOutputIndex = 2;
constexpr int kInputsTensor[] = {2, kShape1Index, kShape2Index};
constexpr int kOutputsTensor[] = {1, kOutputIndex};

// This function is NOT thread safe.
template <typename DimsType>
tflite::micro::KernelRunner CreateBroadcastArgsTestRunner(
    int* input1_shape, DimsType* input1_data, int* input2_shape,
    DimsType* input2_data, int* output_shape, DimsType* output_data) {
  // Some targets do not support dynamic memory (i.e., no malloc or new), thus,
  // the test need to place non-transient memories in static variables. This is
  // safe because tests are guaranteed to run serially.
  // Both below structures are trivially destructible.
  static TFLMRegistration registration;
  static TfLiteTensor tensors[3];

  tensors[0] = CreateTensor(input1_data, IntArrayFromInts(input1_shape));
  tensors[1] = CreateTensor(input2_data, IntArrayFromInts(input2_shape));
  tensors[2] = CreateTensor(output_data, IntArrayFromInts(output_shape));

  registration = tflite::Register_BROADCAST_ARGS();
  tflite::micro::KernelRunner runner = tflite::micro::KernelRunner(
      registration, tensors, sizeof(tensors) / sizeof(TfLiteTensor),
      IntArrayFromInts(const_cast<int*>(kInputsTensor)),
      IntArrayFromInts(const_cast<int*>(kOutputsTensor)),
      /*builtin_data=*/nullptr);
  return runner;
}

template <typename DimsType>
void TestBroadcastArgs(int* input1_shape, DimsType* input1_data,
                       int* input2_shape, DimsType* input2_data,
                       int* output_shape, DimsType* output_data,
                       DimsType* expected_output_data) {
  tflite::micro::KernelRunner runner =
      CreateBroadcastArgsTestRunner(input1_shape, input1_data, input2_shape,
                                    input2_data, output_shape, output_data);

  TF_LITE_MICRO_EXPECT_EQ(runner.InitAndPrepare(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(runner.Invoke(), kTfLiteOk);

  // The output elements contain the fill value.
  const auto elements = tflite::ElementCount(*IntArrayFromInts(output_shape));
  for (int i = 0; i < elements; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(output_data[i], expected_output_data[i]);
  }
}
}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(BroadcastArgsWithScalar) {
  int input1_shape[] = {1, 0};
  int32_t input1_data[] = {};

  int input2_shape[] = {1, 2};
  int32_t input2_data[2] = {2, 4};

  int output_shape[] = {1, 2};
  int32_t output_data[2];
  int32_t expected_output_data[2] = {2, 4};

  TestBroadcastArgs(input1_shape, input1_data, input2_shape, input2_data,
                    output_shape, output_data, expected_output_data);
}

TF_LITE_MICRO_TEST(BroadcastArgsDifferentDims) {
  int input1_shape[] = {1, 1};
  int32_t input1_data[] = {1};

  int input2_shape[] = {1, 2};
  int32_t input2_data[2] = {2, 4};

  int output_shape[] = {1, 2};
  int32_t output_data[2];
  int32_t expected_output_data[2] = {2, 4};

  TestBroadcastArgs(input1_shape, input1_data, input2_shape, input2_data,
                    output_shape, output_data, expected_output_data);
}

TF_LITE_MICRO_TEST(BroadcastArgsSameDims) {
  int input1_shape[] = {1, 6};
  int32_t input1_data[] = {1, 4, 6, 3, 1, 5};

  int input2_shape[] = {1, 6};
  int32_t input2_data[6] = {4, 4, 1, 3, 4, 1};

  int output_shape[] = {1, 6};
  int32_t output_data[6];
  int32_t expected_output_data[6] = {4, 4, 6, 3, 4, 5};

  TestBroadcastArgs(input1_shape, input1_data, input2_shape, input2_data,
                    output_shape, output_data, expected_output_data);
}

TF_LITE_MICRO_TEST(BroadcastArgsComplex) {
  int input1_shape[] = {1, 4};
  int32_t input1_data[] = {6, 3, 1, 5};

  int input2_shape[] = {1, 6};
  int32_t input2_data[6] = {4, 4, 1, 3, 4, 1};

  int output_shape[] = {1, 6};
  int32_t output_data[6];
  int32_t expected_output_data[6] = {4, 4, 6, 3, 4, 5};

  TestBroadcastArgs(input1_shape, input1_data, input2_shape, input2_data,
                    output_shape, output_data, expected_output_data);
}

TF_LITE_MICRO_TESTS_END
