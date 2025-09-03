/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

template <typename T>
void ValidatePackGoldens(TfLiteTensor* tensors, int tensors_size,
                         TfLitePackParams params, TfLiteIntArray* inputs_array,
                         TfLiteIntArray* outputs_array, const T* golden,
                         int output_len, float tolerance, T* output) {
  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output_len; ++i) {
    output[i] = 23;
  }

  const TFLMRegistration registration = Register_PACK();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, reinterpret_cast<void*>(&params));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], tolerance);
  }
}

void TestPackTwoInputsFloat(int* input1_dims_data, const float* input1_data,
                            int* input2_dims_data, const float* input2_data,
                            int axis, int* output_dims_data,
                            const float* expected_output_data,
                            float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(input2_data, input2_dims),
                                        CreateTensor(output_data, output_dims)};

  TfLitePackParams builtin_data = {
      .values_count = 2,
      .axis = axis,
  };
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  ValidatePackGoldens(tensors, tensors_size, builtin_data, inputs_array,
                      outputs_array, expected_output_data, output_dims_count,
                      1e-5f, output_data);
}

void TestPackThreeInputsFloat(int* input1_dims_data, const float* input1_data,
                              int* input2_dims_data, const float* input2_data,
                              int* input3_dims_data, const float* input3_data,
                              int axis, int* output_dims_data,
                              const float* expected_output_data,
                              float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* input3_dims = IntArrayFromInts(input3_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 3;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(input2_data, input2_dims),
                                        CreateTensor(input3_data, input3_dims),
                                        CreateTensor(output_data, output_dims)};

  TfLitePackParams builtin_data = {
      .values_count = 3,
      .axis = axis,
  };
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  ValidatePackGoldens(tensors, tensors_size, builtin_data, inputs_array,
                      outputs_array, expected_output_data, output_dims_count,
                      1e-5f, output_data);
}

template <typename T>
void TestPackTwoInputs(int* input1_dims_data, const T* input1_data,
                       int* input2_dims_data, const T* input2_data, int axis,
                       int* output_dims_data, const T* expected_output_data,
                       T* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(input2_data, input2_dims),
                                        CreateTensor(output_data, output_dims)};

  TfLitePackParams builtin_data = {
      .values_count = 2,
      .axis = axis,
  };
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  ValidatePackGoldens(tensors, tensors_size, builtin_data, inputs_array,
                      outputs_array, expected_output_data, output_dims_count,
                      1e-5f, output_data);
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(PackFloatThreeInputs) {
  int input_shape[] = {1, 2};
  int output_shape[] = {2, 3, 2};
  const float input1_values[] = {1, 4};
  const float input2_values[] = {2, 5};
  const float input3_values[] = {3, 6};
  const float golden[] = {1, 4, 2, 5, 3, 6};
  const int axis = 0;
  constexpr int output_dims_count = 6;
  float output_data[output_dims_count];

  tflite::testing::TestPackThreeInputsFloat(
      input_shape, input1_values, input_shape, input2_values, input_shape,
      input3_values, axis, output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackFloatThreeInputsDifferentAxis) {
  int input_shape[] = {1, 2};
  int output_shape[] = {2, 2, 3};
  const float input1_values[] = {1, 4};
  const float input2_values[] = {2, 5};
  const float input3_values[] = {3, 6};
  const float golden[] = {1, 2, 3, 4, 5, 6};
  const int axis = 1;
  constexpr int output_dims_count = 6;
  float output_data[output_dims_count];

  tflite::testing::TestPackThreeInputsFloat(
      input_shape, input1_values, input_shape, input2_values, input_shape,
      input3_values, axis, output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackFloatThreeInputsNegativeAxis) {
  int input_shape[] = {1, 2};
  int output_shape[] = {2, 2, 3};
  const float input1_values[] = {1, 4};
  const float input2_values[] = {2, 5};
  const float input3_values[] = {3, 6};
  const float golden[] = {1, 2, 3, 4, 5, 6};
  const int axis = -1;
  constexpr int output_dims_count = 6;
  float output_data[output_dims_count];

  tflite::testing::TestPackThreeInputsFloat(
      input_shape, input1_values, input_shape, input2_values, input_shape,
      input3_values, axis, output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackFloatMultiDimensions) {
  int input_shape[] = {2, 2, 3};
  int output_shape[] = {3, 2, 2, 3};
  const float input1_values[] = {1, 2, 3, 4, 5, 6};
  const float input2_values[] = {7, 8, 9, 10, 11, 12};
  const float golden[] = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  const int axis = 1;
  constexpr int output_dims_count = 12;
  float output_data[output_dims_count];

  tflite::testing::TestPackTwoInputsFloat(input_shape, input1_values,
                                          input_shape, input2_values, axis,
                                          output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackInt8MultiDimensions) {
  int input_shape[] = {2, 2, 3};
  int output_shape[] = {3, 2, 2, 3};
  const int8_t input1_values[] = {1, 2, 3, 4, 5, 6};
  const int8_t input2_values[] = {7, 8, 9, 10, 11, 12};
  const int8_t golden[] = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  const int axis = 1;
  constexpr int output_dims_count = 12;
  int8_t output_data[output_dims_count];

  tflite::testing::TestPackTwoInputs<int8_t>(input_shape, input1_values,
                                             input_shape, input2_values, axis,
                                             output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackInt16MultiDimensions) {
  int input_shape[] = {2, 2, 3};
  int output_shape[] = {3, 2, 2, 3};
  const int16_t input1_values[] = {1, 2, 3, 4, 5, 6};
  const int16_t input2_values[] = {7, 8, 9, 10, 11, 12};
  const int16_t golden[] = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  const int axis = 1;
  constexpr int output_dims_count = 12;
  int16_t output_data[output_dims_count];

  tflite::testing::TestPackTwoInputs<int16_t>(
      input_shape, input1_values, input_shape, input2_values, axis,
      output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackInt32MultiDimensions) {
  int input_shape[] = {2, 2, 3};
  int output_shape[] = {3, 2, 2, 3};
  const int32_t input1_values[] = {1, 2, 3, 4, 5, 6};
  const int32_t input2_values[] = {7, 8, 9, 10, 11, 12};
  const int32_t golden[] = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  const int axis = 1;
  constexpr int output_dims_count = 12;
  int32_t output_data[output_dims_count];

  tflite::testing::TestPackTwoInputs<int32_t>(
      input_shape, input1_values, input_shape, input2_values, axis,
      output_shape, golden, output_data);
}

TF_LITE_MICRO_TESTS_END
