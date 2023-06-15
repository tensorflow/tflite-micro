/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename dataT, typename shapeT>
void TestSlice(int* input_dims_data, const dataT* input_data,
               int* begin_dims_data, const shapeT* begin_data,
               int* size_dims_data, const shapeT* size_data,
               int* output_dims_data, const dataT* expected_output_data,
               dataT* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* begin_dims = IntArrayFromInts(begin_dims_data);
  TfLiteIntArray* size_dims = IntArrayFromInts(size_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(begin_data, begin_dims),
      CreateTensor(size_data, size_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_SLICE();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(In1D) {
  int input_shape[] = {1, 4};
  float input_values[] = {1, 2, 3, 4};
  int begin_shape[] = {1, 1};
  int32_t begin_values[] = {1};
  int size_shape[] = {1, 1};
  int32_t size_values[] = {2};
  int output_shape[] = {1, 2};
  float expected_output_data[] = {2, 3};
  float output_data[2];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(In2D) {
  int input_shape[] = {2, 2, 3};
  float input_values[] = {1, 2, 3, 4, 5, 6};
  int begin_shape[] = {1, 2};
  int32_t begin_values[] = {1, 0};
  int size_shape[] = {1, 2};
  int32_t size_values[] = {1, 2};
  int output_shape[] = {1, 2};
  float expected_output_data[] = {4, 5};
  float output_data[2];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(In3D) {
  int input_shape[] = {3, 2, 3, 2};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int begin_shape[] = {1, 3};
  int32_t begin_values[] = {0, 0, 0};
  int size_shape[] = {1, 3};
  int32_t size_values[] = {2, 3, 2};
  int output_shape[] = {3, 2, 3, 2};
  float expected_output_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float output_data[12];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(In5D) {
  int input_shape[] = {5, 5, 1, 1, 1, 1};
  float input_values[] = {1, 2, 3, 4, 5};
  int begin_shape[] = {1, 5};
  int32_t begin_values[] = {1, 0, 0, 0, 0};
  int size_shape[] = {1, 5};
  int32_t size_values[] = {3, 1, 1, 1, 1};
  int output_shape[] = {5, 3, 1, 1, 1, 1};
  float expected_output_data[] = {2, 3, 4};
  float output_data[3];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(InputFloat) {
  int input_shape[] = {4, 4, 1, 1, 1};
  float input_values[] = {1, 2, 3, 4};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 0, 0, 0};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {3, 1, 1, 1};
  int output_shape[] = {4, 3, 1, 1, 1};
  float expected_output_data[] = {2, 3, 4};
  float output_data[3];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(IndexInt64) {
  int input_shape[] = {4, 4, 1, 1, 1};
  float input_values[] = {1, 2, 3, 4};
  int begin_shape[] = {1, 4};
  int64_t begin_values[] = {1, 0, 0, 0};
  int size_shape[] = {1, 4};
  int64_t size_values[] = {3, 1, 1, 1};
  int output_shape[] = {4, 3, 1, 1, 1};
  float expected_output_data[] = {2, 3, 4};
  float output_data[3];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

// See these test cases under:
// https://www.tensorflow.org/versions/master/api_docs/python/tf/slice
TF_LITE_MICRO_TEST(InputInteger1) {
  int input_shape[] = {4, 3, 2, 3, 1};
  int32_t input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                            4, 4, 4, 5, 5, 5, 6, 6, 6};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 0, 0, 0};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {1, 1, 3, 1};
  int output_shape[] = {4, 1, 1, 3, 1};
  int32_t expected_output_data[] = {3, 3, 3};
  int32_t output_data[3];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(InputInteger2) {
  int input_shape[] = {4, 3, 2, 3, 1};
  int32_t input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                            4, 4, 4, 5, 5, 5, 6, 6, 6};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 0, 0, 0};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {1, 2, 3, 1};
  int output_shape[] = {4, 1, 2, 3, 1};
  int32_t expected_output_data[] = {3, 3, 3, 4, 4, 4};
  int32_t output_data[6];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(InputInteger3) {
  int input_shape[] = {4, 3, 2, 3, 1};
  int32_t input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                            4, 4, 4, 5, 5, 5, 6, 6, 6};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 0, 0, 0};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {2, 1, 3, 1};
  int output_shape[] = {4, 2, 1, 3, 1};
  int32_t expected_output_data[] = {3, 3, 3, 5, 5, 5};
  int32_t output_data[6];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(SizeMinus1) {
  int input_shape[] = {4, 3, 2, 3, 1};
  int32_t input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                            4, 4, 4, 5, 5, 5, 6, 6, 6};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 0, 0, 0};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {2, 1, -1, 1};
  int output_shape[] = {4, 2, 1, 3, 1};
  int32_t expected_output_data[] = {3, 3, 3, 5, 5, 5};
  int32_t output_data[6];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(BeginNonZeroSizeMinus1Axis1) {
  int input_shape[] = {4, 3, 3, 2, 1};
  int32_t input_values[] = {1, 1, 2, 2, 3, 3, 4, 4, 5,
                            5, 6, 6, 7, 7, 8, 8, 9, 9};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 1, 0, 0};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {2, -1, 1, 1};
  int output_shape[] = {4, 2, 2, 1, 1};
  int32_t expected_output_data[] = {5, 6, 8, 9};
  int32_t output_data[4];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(BeginNonZeroSizeMinus1Axis2) {
  int input_shape[] = {4, 3, 2, 3, 1};
  int32_t input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                            4, 4, 4, 5, 5, 5, 6, 6, 6};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 0, 1, 0};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {2, 1, -1, 1};
  int output_shape[] = {4, 2, 1, 2, 1};
  int32_t expected_output_data[] = {3, 3, 5, 5};
  int32_t output_data[4];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(BeginNonZeroSizeMinus1Axis3) {
  int input_shape[] = {4, 3, 1, 2, 3};
  int32_t input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                            4, 4, 4, 5, 5, 5, 6, 6, 6};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 0, 0, 1};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {2, 1, 1, -1};
  int output_shape[] = {4, 2, 1, 1, 2};
  int32_t expected_output_data[] = {3, 3, 5, 5};
  int32_t output_data[4];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(SliceInt8) {
  int input_shape[] = {4, 3, 2, 3, 1};
  int8_t input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                           4, 4, 4, 5, 5, 5, 6, 6, 6};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 0, 0, 0};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {2, 1, -1, 1};
  int output_shape[] = {4, 2, 1, 3, 1};
  int8_t expected_output_data[] = {3, 3, 3, 5, 5, 5};
  int8_t output_data[6];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(SliceInt16) {
  int input_shape[] = {4, 3, 2, 3, 1};
  int16_t input_values[] = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                            4, 4, 4, 5, 5, 5, 6, 6, 6};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 0, 0, 0};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {2, 1, -1, 1};
  int output_shape[] = {4, 2, 1, 3, 1};
  int16_t expected_output_data[] = {3, 3, 3, 5, 5, 5};
  int16_t output_data[6];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(SliceBool) {
  int input_shape[] = {4, 3, 2, 3, 1};
  bool input_values[] = {false, false, false, false, false, false,
                         true,  false, true,  false, false, false,
                         false, false, true,  false, false, false};
  int begin_shape[] = {1, 4};
  int32_t begin_values[] = {1, 0, 0, 0};
  int size_shape[] = {1, 4};
  int32_t size_values[] = {2, 1, -1, 1};
  int output_shape[] = {4, 2, 1, 3, 1};
  bool expected_output_data[] = {true, false, true, false, false, true};
  bool output_data[6];

  tflite::testing::TestSlice(input_shape, input_values, begin_shape,
                             begin_values, size_shape, size_values,
                             output_shape, expected_output_data, output_data);
}

TF_LITE_MICRO_TESTS_END
