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
#include <cstdint>
#include <initializer_list>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr int inputs_size = 2;
constexpr int outputs_size = 1;
constexpr int tensors_size = inputs_size + outputs_size;

void TestComparison(const TFLMRegistration& registration, TfLiteTensor* tensors,
                    bool* expected_output_data, bool* output_data) {
  const int output_dims_count = ElementCount(*tensors[inputs_size].dims);

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

void TestComparisonFloat(const TFLMRegistration& registration,
                         int* input1_dims_data, float* input1_data,
                         int* input2_dims_data, float* input2_data,
                         bool* expected_output_data, int* output_dims_data,
                         bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  TestComparison(registration, tensors, expected_output_data, output_data);
}

void TestComparisonBool(const TFLMRegistration& registration,
                        int* input1_dims_data, bool* input1_data,
                        int* input2_dims_data, bool* input2_data,
                        bool* expected_output_data, int* output_dims_data,
                        bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  TestComparison(registration, tensors, expected_output_data, output_data);
}

void TestComparisonInt(const TFLMRegistration& registration,
                       int* input1_dims_data, int32_t* input1_data,
                       int* input2_dims_data, int32_t* input2_data,
                       bool* expected_output_data, int* output_dims_data,
                       bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  TestComparison(registration, tensors, expected_output_data, output_data);
}

void TestComparisonQuantizedInt8(const TFLMRegistration& registration,
                                 int* input1_dims_data, float* input1_data,
                                 int8_t* input1_quantized, float input1_scale,
                                 int input1_zero_point, int* input2_dims_data,
                                 float* input2_data, int8_t* input2_quantized,
                                 float input2_scale, int input2_zero_point,
                                 bool* expected_output_data,
                                 int* output_dims_data, bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_quantized, input1_dims,
                            input1_scale, input1_zero_point),
      CreateQuantizedTensor(input2_data, input2_quantized, input2_dims,
                            input2_scale, input2_zero_point),
      CreateTensor(output_data, output_dims),
  };

  TestComparison(registration, tensors, expected_output_data, output_data);
}

void TestComparisonQuantizedInt16(const TFLMRegistration& registration,
                                  int* input1_dims_data, float* input1_data,
                                  int16_t* input1_quantized, float input1_scale,
                                  int input1_zero_point, int* input2_dims_data,
                                  float* input2_data, int16_t* input2_quantized,
                                  float input2_scale, int input2_zero_point,
                                  bool* expected_output_data,
                                  int* output_dims_data, bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_quantized, input1_dims,
                            input1_scale, input1_zero_point),
      CreateQuantizedTensor(input2_data, input2_quantized, input2_dims,
                            input2_scale, input2_zero_point),
      CreateTensor(output_data, output_dims),
  };

  TestComparison(registration, tensors, expected_output_data, output_data);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(EqualBool) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  bool input1_data[] = {true, false, true, false};
  bool input2_data[] = {true, true, false, false};

  bool expected_data[] = {true, false, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonBool(tflite::Register_EQUAL(), input1_dim,
                                      input1_data, input2_dim, input2_data,
                                      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {true, false, false, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::Register_EQUAL(), input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {false, false, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};
  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::Register_EQUAL(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {false, false, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::Register_EQUAL(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 2, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {false, false, false, false,
                          false, false, true,  false};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(tflite::Register_EQUAL(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualBool) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  bool input1_data[] = {true, false, true, false};
  bool input2_data[] = {true, true, false, false};

  bool expected_data[] = {false, true, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonBool(tflite::Register_NOT_EQUAL(), input1_dim,
                                      input1_data, input2_dim, input2_data,
                                      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {false, true, true, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::Register_NOT_EQUAL(), input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {true, true, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::Register_NOT_EQUAL(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {true, true, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::Register_NOT_EQUAL(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 2, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {true, true, true, true, true, true, false, true};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(tflite::Register_NOT_EQUAL(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {false, true, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::Register_GREATER(), input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {false, true, false, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::Register_GREATER(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {false, true, false, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::Register_GREATER(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 2, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {false, true, true, false, false, true, false, true};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(tflite::Register_GREATER(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {true, true, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::Register_GREATER_EQUAL(), input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {false, true, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::Register_GREATER_EQUAL(), input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {false, true, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::Register_GREATER_EQUAL(), input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 2, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {false, true, true, false, false, true, true, true};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(
      tflite::Register_GREATER_EQUAL(), input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {false, false, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::Register_LESS(), input1_dim, input1_data, input2_dim, input2_data,
      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 6, 5};

  bool expected_data[] = {true, false, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::Register_LESS(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {true, false, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::Register_LESS(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 6, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {true, false, false, true, true, false, false, false};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(tflite::Register_LESS(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {true, false, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::Register_LESS_EQUAL(), input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {true, false, true, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::Register_LESS_EQUAL(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {true, false, true, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::Register_LESS_EQUAL(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 2, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {true, false, false, true, true, false, true, false};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(tflite::Register_LESS_EQUAL(), input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualQuantizedInt8) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};

  float input1_data[] = {1, -9, 7, 3};
  float input2_data[] = {-1, 2, 7, 5};

  bool expected_data[] = {false, false, true, false};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = -5;
  const float input2_scale = 0.25;
  const int input2_zero_point = 5;
  int8_t input1_quantized[4];
  int8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedInt8(
      tflite::Register_EQUAL(), input1_dim, input1_data, input1_quantized,
      input1_scale, input1_zero_point, input2_dim, input2_data,
      input2_quantized, input2_scale, input2_zero_point, expected_data,
      expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualQuantizedInt8) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};

  float input1_data[] = {1, -9, 7, 3};
  float input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {false, true, false, true};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = -5;
  const float input2_scale = 0.25;
  const int input2_zero_point = 5;
  int8_t input1_quantized[4];
  int8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedInt8(
      tflite::Register_NOT_EQUAL(), input1_dim, input1_data, input1_quantized,
      input1_scale, input1_zero_point, input2_dim, input2_data,
      input2_quantized, input2_scale, input2_zero_point, expected_data,
      expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualQuantizedInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {true, true, true, false, true, true};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    int8_t input1_quantized[6];
    int8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::Register_NOT_EQUAL(), input1_dim, input1_data, input1_quantized,
        input1_scale, input1_zero_point, input2_dim, input2_data,
        input2_quantized, input1_scale, input1_zero_point, expected_data,
        expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterQuantizedInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {true, false, false, false, true, true};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    int8_t input1_quantized[6];
    int8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::Register_GREATER(), input1_dim, input1_data, input1_quantized,
        input1_scale, input1_zero_point, input2_dim, input2_data,
        input2_quantized, input1_scale, input1_zero_point, expected_data,
        expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterQuantizedInt16WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {true, false, false, false, true, true};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    int16_t input1_quantized[6];
    int16_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt16(
        tflite::Register_GREATER(), input1_dim, input1_data, input1_quantized,
        input1_scale, input1_zero_point, input2_dim, input2_data,
        input2_quantized, input1_scale, input1_zero_point, expected_data,
        expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterEqualQuantizedInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {true, false, false, true, true, true};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    int8_t input1_quantized[6];
    int8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::Register_GREATER_EQUAL(), input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(LessQuantizedInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {false, true, true, false, false, false};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    int8_t input1_quantized[6];
    int8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::Register_LESS(), input1_dim, input1_data, input1_quantized,
        input1_scale, input1_zero_point, input2_dim, input2_data,
        input2_quantized, input1_scale, input1_zero_point, expected_data,
        expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(LessEqualQuantizedInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {false, true, true, true, false, false};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    int8_t input1_quantized[6];
    int8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::Register_LESS_EQUAL(), input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TESTS_END
