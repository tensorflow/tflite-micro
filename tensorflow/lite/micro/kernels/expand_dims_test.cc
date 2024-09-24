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

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// The tensor layout is fixed.
constexpr int kInputsTensorSize = 2;
constexpr int kOutputsTensorSize = 1;
constexpr int kTensorsSize = kInputsTensorSize + kOutputsTensorSize;

constexpr int kDimsTensorIndex = 0;
constexpr int kAxisTensorIndex = 1;
constexpr int kOutputTensorIndex = 2;
constexpr int kInputTensors[] = {2, kDimsTensorIndex, kAxisTensorIndex};
constexpr int kOutputTensors[] = {1, kOutputTensorIndex};

template <typename T>
micro::KernelRunner CreateExpandDimsKernelRunner(
    int* input_dims, const T* input_data, int* axis_dims,
    const int32_t* axis_data, int* output_dims, T* output_data) {
  // Some targets do not support dynamic memory (i.e., no malloc or new), thus,
  // the test need to place non-transitent memories in static variables. This is
  // safe because tests are guaranteed to run serially.
  // Both below structures are trivially destructible.
  static TFLMRegistration registration;
  static TfLiteTensor tensors[kTensorsSize];

  TfLiteIntArray* in_dims = IntArrayFromInts(input_dims);
  TfLiteIntArray* ax_dims = IntArrayFromInts(axis_dims);
  TfLiteIntArray* out_dims = IntArrayFromInts(output_dims);

  const int out_dims_size = out_dims->size;
  const int in_dims_size = in_dims->size;
  TF_LITE_MICRO_EXPECT_EQ(out_dims_size, (in_dims_size + 1));

  tensors[kDimsTensorIndex] = CreateTensor(input_data, in_dims);
  tensors[kAxisTensorIndex] = CreateTensor(axis_data, ax_dims);
  tensors[kOutputTensorIndex] = CreateTensor(output_data, out_dims, true);

  TfLiteIntArray* inputs_array =
      IntArrayFromInts(const_cast<int*>(kInputTensors));

  TfLiteIntArray* outputs_array =
      IntArrayFromInts(const_cast<int*>(kOutputTensors));

  registration = Register_EXPAND_DIMS();
  micro::KernelRunner runner(registration, tensors, kTensorsSize, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);
  return runner;
}

template <typename T>
void TestExpandDims(int* input_dims, const T* input_data, int* axis_dims,
                    const int32_t* axis_data, int* expected_output_dims,
                    int* output_dims, const T* expected_output_data,
                    T* output_data) {
  micro::KernelRunner runner = CreateExpandDimsKernelRunner(
      input_dims, input_data, axis_dims, axis_data, output_dims, output_data);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  // The output tensor's data have been updated by the kernel.
  TfLiteIntArray* actual_out_dims = IntArrayFromInts(output_dims);
  const int output_size = ElementCount(*actual_out_dims);

  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ExpandDimsPositiveAxisTest0) {
  int8_t output_data[4];
  int input_dims[] = {2, 2, 2};
  const int8_t input_data[] = {-1, 1, -2, 2};
  const int8_t golden_data[] = {-1, 1, -2, 2};
  int axis_dims[] = {1, 1};
  const int32_t axis_data[] = {0};
  int golden_dims[] = {1, 2, 2};
  int output_dims[] = {3, 1, 2, 2};
  tflite::testing::TestExpandDims<int8_t>(input_dims, input_data, axis_dims,
                                          axis_data, golden_dims, output_dims,
                                          golden_data, output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsPositiveAxisTest1) {
  float output_data[4];
  int input_dims[] = {2, 2, 2};
  const float input_data[] = {-1.1, 1.2, -2.1, 2.2};
  const float golden_data[] = {-1.1, 1.2, -2.1, 2.2};
  int axis_dims[] = {1, 1};
  const int32_t axis_data[] = {1};
  int golden_dims[] = {2, 1, 2};
  int output_dims[] = {3, 2, 1, 2};
  tflite::testing::TestExpandDims<float>(input_dims, input_data, axis_dims,
                                         axis_data, golden_dims, output_dims,
                                         golden_data, output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsPositiveAxisTest2) {
  int8_t output_data[4];
  int input_dims[] = {2, 2, 2};
  const int8_t input_data[] = {-1, 1, -2, 2};
  const int8_t golden_data[] = {-1, 1, -2, 2};
  int axis_dims[] = {1, 1};
  const int32_t axis_data[] = {2};
  int golden_dims[] = {2, 2, 1};
  int output_dims[] = {3, 2, 2, 1};
  tflite::testing::TestExpandDims<int8_t>(input_dims, input_data, axis_dims,
                                          axis_data, golden_dims, output_dims,
                                          golden_data, output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsPositiveAxisTest3) {
  int16_t output_data[6];
  int input_dims[] = {3, 3, 1, 2};
  const int16_t input_data[] = {-1, 1, 2, -2, 0, 3};
  const int16_t golden_data[] = {-1, 1, 2, -2, 0, 3};
  int axis_dims[] = {1, 1};
  const int32_t axis_data[] = {3};
  int golden_dims[] = {1, 3, 1, 2};
  int output_dims[] = {4, 3, 1, 2, 1};
  tflite::testing::TestExpandDims<int16_t>(input_dims, input_data, axis_dims,
                                           axis_data, golden_dims, output_dims,
                                           golden_data, output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsNegativeAxisTest4) {
  int8_t output_data[6];
  int input_dims[] = {3, 3, 1, 2};
  const int8_t input_data[] = {-1, 1, 2, -2, 0, 3};
  const int8_t golden_data[] = {-1, 1, 2, -2, 0, 3};
  int axis_dims[] = {1, 1};
  const int32_t axis_data[] = {-4};
  int golden_dims[] = {1, 3, 1, 2};
  int output_dims[] = {4, 1, 3, 1, 2};
  tflite::testing::TestExpandDims<int8_t>(input_dims, input_data, axis_dims,
                                          axis_data, golden_dims, output_dims,
                                          golden_data, output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsNegativeAxisTest3) {
  float output_data[6];
  int input_dims[] = {3, 3, 1, 2};
  const float input_data[] = {0.1, -0.8, -1.2, -0.5, 0.9, 1.3};
  const float golden_data[] = {0.1, -0.8, -1.2, -0.5, 0.9, 1.3};
  int axis_dims[] = {1, 1};
  const int32_t axis_data[] = {-3};
  int golden_dims[] = {3, 1, 1, 2};
  int output_dims[] = {4, 3, 1, 1, 2};
  tflite::testing::TestExpandDims<float>(input_dims, input_data, axis_dims,
                                         axis_data, golden_dims, output_dims,
                                         golden_data, output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsNegativeAxisTest2) {
  int8_t output_data[6];
  int input_dims[] = {3, 1, 2, 3};
  const int8_t input_data[] = {-1, 1, 2, -2, 0, 3};
  const int8_t golden_data[] = {-1, 1, 2, -2, 0, 3};
  int axis_dims[] = {1, 1};
  const int32_t axis_data[] = {-2};
  int golden_dims[] = {1, 2, 1, 3};
  int output_dims[] = {4, 1, 2, 1, 3};
  tflite::testing::TestExpandDims<int8_t>(input_dims, input_data, axis_dims,
                                          axis_data, golden_dims, output_dims,
                                          golden_data, output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsNegativeAxisTest1) {
  float output_data[6];
  int input_dims[] = {3, 1, 3, 2};
  const float input_data[] = {0.1, -0.8, -1.2, -0.5, 0.9, 1.3};
  const float golden_data[] = {0.1, -0.8, -1.2, -0.5, 0.9, 1.3};
  int axis_dims[] = {1, 1};
  const int32_t axis_data[] = {-1};
  int golden_dims[] = {1, 3, 2, 1};
  int output_dims[] = {4, 1, 3, 2, 1};
  tflite::testing::TestExpandDims<float>(input_dims, input_data, axis_dims,
                                         axis_data, golden_dims, output_dims,
                                         golden_data, output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsInputOutputDimsMismatchShallFail) {
  float output_data[6];
  int input_dims[] = {3, 1, 3, 2};
  const float input_data[] = {0.1, -0.8, -1.2, -0.5, 0.9, 1.3};
  int axis_dims[] = {1, 1};
  const int32_t axis_data[] = {-1};
  // When input dimension is [1, 3, 2] and the axis is -1, the output dimension
  // should be [1, 3, 2, 1] as in the test case ExpandDimsNegativeAxisTest1.
  // Shuffle the output dimension to make it incorrect so that the EXPAND_DIMS
  // op would fail at prepare.
  int output_dims[] = {4, 1, 3, 1, 2};

  tflite::micro::KernelRunner runner =
      tflite::testing::CreateExpandDimsKernelRunner(input_dims, input_data,
                                                    axis_dims, axis_data,
                                                    output_dims, output_data);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError, runner.InitAndPrepare());
}

TF_LITE_MICRO_TEST(ExpandDimsAxisOutOfRangeShallFail) {
  int8_t output_data[6];
  int input_dims[] = {3, 1, 3, 2};
  const int8_t input_data[] = {1, 8, 2, 5, 9, 3};
  int axis_dims[] = {1, 1};
  // The input dimension is 3-D, so that axis value should not exceed 3.
  // The below axis value 4 shall lead to failure at prepare.
  const int32_t axis_data[] = {4};
  int output_dims[] = {4, 1, 3, 2, 1};

  tflite::micro::KernelRunner runner =
      tflite::testing::CreateExpandDimsKernelRunner(input_dims, input_data,
                                                    axis_dims, axis_data,
                                                    output_dims, output_data);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError, runner.InitAndPrepare());
}

TF_LITE_MICRO_TESTS_END
