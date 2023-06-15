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
constexpr int kInputIndex = 0;
constexpr int kShapeIndex = 1;
constexpr int kOutputIndex = 2;
constexpr int kInputsTensor[] = {2, kInputIndex, kShapeIndex};
constexpr int kOutputsTensor[] = {1, kOutputIndex};

// This function is NOT thread safe.
template <typename DimsType, typename ValueType>
tflite::micro::KernelRunner CreateBroadcastToTestRunner(
    int* dims_shape, DimsType* dims_data, int* input_shape,
    ValueType* input_data, int* output_shape, ValueType* output_data) {
  // Some targets do not support dynamic memory (i.e., no malloc or new), thus,
  // the test need to place non-transient memories in static variables. This is
  // safe because tests are guaranteed to run serially.
  // Both below structures are trivially destructible.
  static TFLMRegistration registration;
  static TfLiteTensor tensors[3];

  tensors[0] = CreateTensor(input_data, IntArrayFromInts(input_shape));
  tensors[1] = CreateTensor(dims_data, IntArrayFromInts(dims_shape));
  tensors[2] = CreateTensor(output_data, IntArrayFromInts(output_shape));

  // The output type matches the value type.
  TF_LITE_MICRO_EXPECT_EQ(tensors[kOutputIndex].type,
                          tensors[kInputIndex].type);

  registration = tflite::Register_BROADCAST_TO();
  tflite::micro::KernelRunner runner = tflite::micro::KernelRunner(
      registration, tensors, sizeof(tensors) / sizeof(TfLiteTensor),
      IntArrayFromInts(const_cast<int*>(kInputsTensor)),
      IntArrayFromInts(const_cast<int*>(kOutputsTensor)),
      /*builtin_data=*/nullptr);
  return runner;
}

template <typename DimsType, typename ValueType>
void TestBroadcastTo(int* dims_shape, DimsType* dims_data, int* input_shape,
                     ValueType* input_data, int* output_shape,
                     ValueType* output_data, ValueType* expected_output_data) {
  tflite::micro::KernelRunner runner =
      CreateBroadcastToTestRunner(dims_shape, dims_data, input_shape,
                                  input_data, output_shape, output_data);

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

TF_LITE_MICRO_TEST(ShapeMustBe1D) {
  int dims_shape[] = {2, 2, 2};
  int32_t dims_data[] = {2, 3, 4, 4};

  int input_shape[] = {2, 2, 2};
  int input_data[] = {2, 3, 4, 4};

  int output_shape[] = {2, 2, 2};
  int output_data[] = {2, 3, 4, 4};

  tflite::micro::KernelRunner runner =
      CreateBroadcastToTestRunner(dims_shape, dims_data, input_shape,
                                  input_data, output_shape, output_data);

  TF_LITE_MICRO_EXPECT_EQ(runner.InitAndPrepare(), kTfLiteError);
}

TF_LITE_MICRO_TEST(TooManyDimensionShouldFail) {
  int dims_shape[] = {1, 6};
  int32_t dims_data[] = {2, 2, 2, 2, 2, 2};

  int input_shape[] = {2, 2, 2};
  int input_data[] = {2, 3, 4, 4};

  int output_shape[] = {6, 2, 2, 2, 2, 2, 2};
  int output_data[12];

  tflite::micro::KernelRunner runner =
      CreateBroadcastToTestRunner(dims_shape, dims_data, input_shape,
                                  input_data, output_shape, output_data);

  TF_LITE_MICRO_EXPECT_EQ(runner.InitAndPrepare(), kTfLiteError);
}

TF_LITE_MICRO_TEST(MismatchDimensionShouldFail) {
  int dims_shape[] = {1, 4};
  int32_t dims_data[] = {2, 4, 1, 3};

  int input_shape[] = {2, 4, 1, 3};
  int input_data[24] = {2, 3, 4, 4};

  int output_shape[] = {4, 2, 4, 1, 2};
  int output_data[24];

  tflite::micro::KernelRunner runner =
      CreateBroadcastToTestRunner(dims_shape, dims_data, input_shape,
                                  input_data, output_shape, output_data);

  TF_LITE_MICRO_EXPECT_EQ(runner.InitAndPrepare(), kTfLiteError);
}

TF_LITE_MICRO_TEST(Broadcast1DConstTest) {
  constexpr int kDimension = 4;
  constexpr int kSize = 4;
  int dims_shape[] = {1, 1};
  int32_t dims_data[] = {kDimension};

  int input_shape[] = {1, 1};
  int32_t input_data[] = {3};

  int output_shape[] = {1, kDimension};
  int32_t output_data[kSize];
  int32_t expected_output_data[kSize] = {3, 3, 3, 3};

  TestBroadcastTo(dims_shape, dims_data, input_shape, input_data, output_shape,
                  output_data, expected_output_data);
}

TF_LITE_MICRO_TEST(Broadcast4DConstTest) {
  int dims_shape[] = {1, 4};
  int32_t dims_data[] = {2, 2, 2, 2};

  int input_shape[] = {2, 2, 2};
  int32_t input_data[4] = {2, 3, 4, 5};

  int output_shape[] = {4, 2, 2, 2, 2};
  int32_t output_data[16];
  int32_t expected_output_data[16] = {2, 3, 4, 5, 2, 3, 4, 5,
                                      2, 3, 4, 5, 2, 3, 4, 5};

  TestBroadcastTo(dims_shape, dims_data, input_shape, input_data, output_shape,
                  output_data, expected_output_data);
}

TF_LITE_MICRO_TEST(ComplexBroadcast4DConstTest) {
  int dims_shape[] = {1, 4};
  int32_t dims_data[] = {3, 3, 2, 2};

  int input_shape[] = {4, 1, 3, 1, 2};
  int32_t input_data[6] = {1, 2, 3, 4, 5, 6};

  int output_shape[] = {4, 3, 3, 2, 2};
  int32_t output_data[36];
  int32_t expected_output_data[36] = {1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6,
                                      1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6,
                                      1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6};

  TestBroadcastTo(dims_shape, dims_data, input_shape, input_data, output_shape,
                  output_data, expected_output_data);
}

TF_LITE_MICRO_TEST(NoBroadcastingConstTest) {
  int dims_shape[] = {1, 3};
  int32_t dims_data[] = {3, 1, 2};

  int input_shape[] = {3, 3, 1, 2};
  int32_t input_data[6] = {1, 2, 3, 4, 5, 6};

  int output_shape[] = {3, 3, 1, 2};
  int32_t output_data[6];
  int32_t expected_output_data[6] = {1, 2, 3, 4, 5, 6};

  TestBroadcastTo(dims_shape, dims_data, input_shape, input_data, output_shape,
                  output_data, expected_output_data);
}

TF_LITE_MICRO_TEST(BroadcastInt64ShapeTest) {
  int dims_shape[] = {1, 4};
  int64_t dims_data[] = {1, 1, 2, 2};

  int input_shape[] = {4, 1, 1, 1, 2};
  int32_t input_data[2] = {3, 4};

  int output_shape[] = {4, 1, 1, 2, 2};
  int32_t output_data[4];
  int32_t expected_output_data[4] = {3, 4, 3, 4};

  TestBroadcastTo(dims_shape, dims_data, input_shape, input_data, output_shape,
                  output_data, expected_output_data);
}

TF_LITE_MICRO_TESTS_END
