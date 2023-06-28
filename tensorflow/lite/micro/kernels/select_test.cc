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

#include <type_traits>

#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

typedef struct {
  EmptyStructPlaceholder placeholder;
} TfLiteSelectParams;

template <typename T>
void TestSelect(int* input1_dims_data, const bool* input1_data,
                int* input2_dims_data, const T* input2_data,
                int* input3_dims_data, const T* input3_data,
                int* output_dims_data, T* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* input3_dims = IntArrayFromInts(input3_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 3;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(input2_data, input2_dims),
                                        CreateTensor(input3_data, input3_dims),
                                        CreateTensor(output_data, output_dims)};

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteSelectParams builtin_data;
  const TFLMRegistration registration = tflite::Register_SELECT_V2();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void ExpectEqual(int* dims, const T* expected_data, const T* output_data) {
  TfLiteIntArray* dims_array = IntArrayFromInts(dims);
  const int element_count = ElementCount(*dims_array);
  for (int i = 0; i < element_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_data[i], output_data[i]);
  }
}

template <typename T>
void ExpectNear(int* dims, const T* expected_data, const T* output_data) {
  TfLiteIntArray* dims_array = IntArrayFromInts(dims);
  const int element_count = ElementCount(*dims_array);
  for (int i = 0; i < element_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], 1e-5f);
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SelectFloat) {
  int inout_shape[] = {4, 1, 1, 1, 4};

  const bool input1_data[] = {true, false, true, false};
  const float input2_data[] = {0.1f, 0.2f, 0.3f, 0.4f};
  const float input3_data[] = {0.5f, 0.6f, 0.7f, 0.8f};
  const float expected_output[] = {0.1f, 0.6f, 0.3, 0.8f};

  float output_data[4];
  tflite::testing::TestSelect(inout_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectNear(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(SelectInt8) {
  int inout_shape[] = {4, 1, 1, 1, 4};

  const bool input1_data[] = {false, true, false, false};
  const int8_t input2_data[] = {1, -2, 3, 4};
  const int8_t input3_data[] = {5, 6, 7, -8};
  const int8_t expected_output[] = {5, -2, 7, -8};

  int8_t output_data[4];
  tflite::testing::TestSelect(inout_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(SelectInt16) {
  int inout_shape[] = {4, 1, 1, 1, 4};

  const bool input1_data[] = {false, true, false, false};
  const int16_t input2_data[] = {1, 2, 3, 4};
  const int16_t input3_data[] = {5, 6, 7, 8};
  const int16_t expected_output[] = {5, 2, 7, 8};

  int16_t output_data[4];
  tflite::testing::TestSelect(inout_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(BroadcastSelectInt16OneDimensionConditionWithSingleValue) {
  int input1_shape[] = {1, 1};
  int input2_shape[] = {5, 1, 2, 2, 2, 1};
  int input3_shape[] = {4, 1, 2, 2, 1};

  const bool input1_data[] = {false};
  const int16_t input2_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const int16_t input3_data[] = {9, 10, 11, 12};
  const int16_t expected_output[] = {9, 10, 11, 12, 9, 10, 11, 12};

  int16_t output_data[8];
  tflite::testing::TestSelect(input1_shape, input1_data, input2_shape,
                              input2_data, input3_shape, input3_data,
                              input2_shape, output_data);
  tflite::testing::ExpectEqual(input2_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(BroadcastSelectInt16LesserThan4D) {
  int input1_shape[] = {2, 1, 2};
  int inout_shape[] = {3, 1, 2, 2};

  const bool input1_data[] = {false, true};
  const int16_t input2_data[] = {1, 2, 3, 4};
  const int16_t input3_data[] = {5, 6, 7, 8};
  const int16_t expected_output[] = {5, 2, 7, 4};

  int16_t output_data[4];
  tflite::testing::TestSelect(input1_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(BroadcastSelectInt16OnFalseValue) {
  int input1_shape[] = {1, 1};
  int inout_shape[] = {3, 1, 2, 2};

  const bool input1_data[] = {false};
  const int16_t input2_data[] = {1, 2, 3, 4};
  const int16_t input3_data[] = {5, 6, 7, 8};
  const int16_t expected_output[] = {5, 6, 7, 8};

  int16_t output_data[4];
  tflite::testing::TestSelect(input1_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(BroadcastSelectInt16) {
  int input1_shape[] = {2, 1, 2};
  int inout_shape[] = {3, 1, 2, 2};

  const bool input1_data[] = {false, true};
  const int16_t input2_data[] = {1, 2, 3, 4};
  const int16_t input3_data[] = {5, 6, 7, 7};
  const int16_t expected_output[] = {5, 2, 7, 4};

  int16_t output_data[4];
  tflite::testing::TestSelect(input1_shape, input1_data, inout_shape,
                              input2_data, inout_shape, input3_data,
                              inout_shape, output_data);
  tflite::testing::ExpectEqual(inout_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(BroadcastSelectInt16OneDimensionConditionWithTwoValues) {
  int input1_shape[] = {1, 2};
  int input_shape[] = {4, 2, 1, 2, 1};
  int output_shape[] = {4, 2, 1, 2, 2};

  const bool input1_data[] = {false, true};
  const int16_t input2_data[] = {1, 2, 3, 4};
  const int16_t input3_data[] = {5, 6, 7, 8};
  const int16_t expected_output[] = {5, 1, 6, 2, 7, 3, 8, 4};

  int16_t output_data[8];
  tflite::testing::TestSelect(input1_shape, input1_data, input_shape,
                              input2_data, input_shape, input3_data,
                              output_shape, output_data);
  tflite::testing::ExpectEqual(output_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(MixedFlatSizeOneInputsWithScalarInputConditionTensor) {
  int input1_shape[] = {0};  // conditional data is a scalar
  int input_shape[] = {1, 1};
  int output_shape[] = {0};  // output data is a scalar

  const bool input1_data[] = {false};
  const int16_t input2_data[] = {1};
  const int16_t input3_data[] = {5};
  const int16_t expected_output[] = {5};

  int16_t output_data[std::extent<decltype(expected_output)>::value];
  tflite::testing::TestSelect(input1_shape, input1_data, input_shape,
                              input2_data, input_shape, input3_data,
                              output_shape, output_data);
  tflite::testing::ExpectEqual(output_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(MixedFlatSizeOneInputsWithScalarInputXTensor) {
  int input2_shape[] = {0};  // x data is a scalar
  int input_shape[] = {1, 1};
  int output_shape[] = {0};  // output data is a scalar

  const bool input1_data[] = {true};
  const int16_t input2_data[] = {1};
  const int16_t input3_data[] = {5};
  const int16_t expected_output[] = {1};

  int16_t output_data[std::extent<decltype(expected_output)>::value];
  tflite::testing::TestSelect(input_shape, input1_data, input2_shape,
                              input2_data, input_shape, input3_data,
                              output_shape, output_data);
  tflite::testing::ExpectEqual(output_shape, expected_output, output_data);
}

TF_LITE_MICRO_TEST(MixedFlatSizeOneInputsWithScalarInputYTensor) {
  int input3_shape[] = {0};  // y data is a scalar
  int input_shape[] = {1, 1};
  int output_shape[] = {0};  // output data is a scalar

  const bool input1_data[] = {false};
  const int16_t input2_data[] = {1};
  const int16_t input3_data[] = {5};
  const int16_t expected_output[] = {5};

  int16_t output_data[std::extent<decltype(expected_output)>::value];
  tflite::testing::TestSelect(input_shape, input1_data, input_shape,
                              input2_data, input3_shape, input3_data,
                              output_shape, output_data);
  tflite::testing::ExpectEqual(output_shape, expected_output, output_data);
}

TF_LITE_MICRO_TESTS_END
