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

void TestUnpackThreeOutputsFloat(
    int* input_dims_data, const float* input_data, int axis,
    int* output1_dims_data, const float* expected_output1_data,
    int* output2_dims_data, const float* expected_output2_data,
    int* output3_dims_data, const float* expected_output3_data,
    float* output1_data, float* output2_data, float* output3_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInts(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInts(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInts(output3_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);
  const int output3_dims_count = ElementCount(*output3_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 3;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output1_data, output1_dims),
      CreateTensor(output2_data, output2_dims),
      CreateTensor(output3_data, output3_dims)};

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  for (int i = 0; i < output3_dims_count; ++i) {
    output3_data[i] = 23;
  }

  TfLiteUnpackParams builtin_data = {
      .num = 3,
      .axis = axis,
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {3, 1, 2, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_UNPACK();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output1_data[i], output1_data[i], 1e-5f);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output2_data[i], output2_data[i], 1e-5f);
  }

  for (int i = 0; i < output3_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output3_data[i], output3_data[i], 1e-5f);
  }
}

void TestUnpackOneOutputFloat(int* input_dims_data, const float* input_data,
                              int axis, int* output_dims_data,
                              const float* expected_output_data,
                              float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input_data, input_dims),
                                        CreateTensor(output_data, output_dims)};

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output_dims_count; ++i) {
    output_data[i] = 23;
  }

  TfLiteUnpackParams builtin_data = {
      .num = 1,
      .axis = axis,
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_UNPACK();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

template <typename T>
void TestUnpackThreeOutputs(int* input_dims_data, const T* input_data, int axis,
                            int* output1_dims_data,
                            const T* expected_output1_data,
                            int* output2_dims_data,
                            const T* expected_output2_data,
                            int* output3_dims_data,
                            const T* expected_output3_data, T* output1_data,
                            T* output2_data, T* output3_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInts(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInts(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInts(output3_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);
  const int output3_dims_count = ElementCount(*output3_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 3;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output1_data, output1_dims),
      CreateTensor(output2_data, output2_dims),
      CreateTensor(output3_data, output3_dims)};

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  for (int i = 0; i < output3_dims_count; ++i) {
    output3_data[i] = 23;
  }

  TfLiteUnpackParams builtin_data = {
      .num = 3,
      .axis = axis,
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {3, 1, 2, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_UNPACK();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output1_data[i], output1_data[i]);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output2_data[i], output2_data[i]);
  }

  for (int i = 0; i < output3_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output3_data[i], output3_data[i]);
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(UnpackFloatThreeOutputs) {
  int input_shape[] = {2, 3, 2};
  const float input_values[] = {1, 2, 3, 4, 5, 6};
  int output1_shape[] = {1, 2};
  const float output1_golden[] = {1, 2};
  int output2_shape[] = {1, 2};
  const float output2_golden[] = {3, 4};
  int output3_shape[] = {1, 2};
  const float output3_golden[] = {5, 6};
  constexpr int output1_dims_count = 2;
  constexpr int output2_dims_count = 2;
  constexpr int output3_dims_count = 2;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  tflite::testing::TestUnpackThreeOutputsFloat(
      input_shape, input_values, 0, output1_shape, output1_golden,
      output2_shape, output2_golden, output3_shape, output3_golden,
      output1_data, output2_data, output3_data);
}

TF_LITE_MICRO_TEST(UnpackFloatThreeOutputsNegativeAxisTwo) {
  int input_shape[] = {2, 3, 2};
  const float input_values[] = {1, 2, 3, 4, 5, 6};
  int output1_shape[] = {1, 2};
  const float output1_golden[] = {1, 2};
  int output2_shape[] = {1, 2};
  const float output2_golden[] = {3, 4};
  int output3_shape[] = {1, 2};
  const float output3_golden[] = {5, 6};
  constexpr int output1_dims_count = 2;
  constexpr int output2_dims_count = 2;
  constexpr int output3_dims_count = 2;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  tflite::testing::TestUnpackThreeOutputsFloat(
      input_shape, input_values, -2, output1_shape, output1_golden,
      output2_shape, output2_golden, output3_shape, output3_golden,
      output1_data, output2_data, output3_data);
}

TF_LITE_MICRO_TEST(UnpackFloatOneOutput) {
  int input_shape[] = {2, 1, 6};
  const float input_values[] = {1, 2, 3, 4, 5, 6};
  int output_shape[] = {1, 6};
  const float golden[] = {1, 2, 3, 4, 5, 6};
  constexpr int output_dims_count = 6;
  float output_data[output_dims_count];
  tflite::testing::TestUnpackOneOutputFloat(input_shape, input_values, 0,
                                            output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(UnpackInt16ThreeOutputs) {
  int input_shape[] = {2, 3, 2};
  const int16_t input_values[] = {1, 2, 3, 4, 5, 6};
  int output1_shape[] = {1, 2};
  const int16_t output1_golden[] = {1, 2};
  int output2_shape[] = {1, 2};
  const int16_t output2_golden[] = {3, 4};
  int output3_shape[] = {1, 2};
  const int16_t output3_golden[] = {5, 6};
  constexpr int output1_dims_count = 2;
  constexpr int output2_dims_count = 2;
  constexpr int output3_dims_count = 2;
  int16_t output1_data[output1_dims_count];
  int16_t output2_data[output2_dims_count];
  int16_t output3_data[output3_dims_count];
  tflite::testing::TestUnpackThreeOutputs<int16_t>(
      input_shape, input_values, 0, output1_shape, output1_golden,
      output2_shape, output2_golden, output3_shape, output3_golden,
      output1_data, output2_data, output3_data);
}

TF_LITE_MICRO_TEST(UnpackInt32ThreeOutputs) {
  int input_shape[] = {2, 3, 2};
  const int32_t input_values[] = {1, 2, 3, 4, 5, 6};
  int output1_shape[] = {1, 2};
  const int32_t output1_golden[] = {1, 2};
  int output2_shape[] = {1, 2};
  const int32_t output2_golden[] = {3, 4};
  int output3_shape[] = {1, 2};
  const int32_t output3_golden[] = {5, 6};
  constexpr int output1_dims_count = 2;
  constexpr int output2_dims_count = 2;
  constexpr int output3_dims_count = 2;
  int32_t output1_data[output1_dims_count];
  int32_t output2_data[output2_dims_count];
  int32_t output3_data[output3_dims_count];
  tflite::testing::TestUnpackThreeOutputs<int32_t>(
      input_shape, input_values, 0, output1_shape, output1_golden,
      output2_shape, output2_golden, output3_shape, output3_golden,
      output1_data, output2_data, output3_data);
}

TF_LITE_MICRO_TESTS_END
