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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

const float kQuantizedTolerance = 2 * (1. / 256);

void ExecutePReluTest(const int tensors_count, TfLiteTensor* tensors) {
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_PRELU();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

void TestPreluFloat(int* input_dims_data, const float* input_data,
                    int* alpha_dims_data, const float* alpha_data,
                    const float* expected_output_data, int* output_dims_data,
                    float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* alpha_dims = IntArrayFromInts(alpha_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(alpha_data, alpha_dims),
      CreateTensor(output_data, output_dims),
  };

  ExecutePReluTest(tensors_size, tensors);

  for (int i = 0; i < output_dims_count; i++) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

template <typename T, typename Slope>
void TestPreluQuantized(int* input_dims_data, const float* input_data,
                        T* input_quantized, const float input_scale,
                        const int input_zero_point, int* alpha_dims_data,
                        const float* alpha_data, Slope* alpha_quantized,
                        const float alpha_scale, const int alpha_zero_point,
                        const float* golden, const float output_scale,
                        const int output_zero_point, int* output_dims_data,
                        T* output_quantized, float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* alpha_dims = IntArrayFromInts(alpha_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(alpha_data, alpha_quantized, alpha_dims,
                            alpha_scale, alpha_zero_point),
      CreateQuantizedTensor(output_quantized, output_dims, output_scale,
                            output_zero_point),
  };

  ExecutePReluTest(tensors_size, tensors);

  Dequantize(output_quantized, output_dims_count, output_scale,
             output_zero_point, output_data);

  for (int i = 0; i < output_dims_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], kQuantizedTolerance);
  }
}
}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatPreluActivationsOpTest) {
  int input_shape[] = {3, 2, 2, 3};
  const float input_values[] = {
      0.0f,  0.0f,  0.0f,   // Row 1, Column 1
      1.0f,  1.0f,  1.0f,   // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,  // Row 2, Column 1
      -2.0f, -2.0f, -2.0f,  // Row 1, Column 2
  };
  int alpha_shape[] = {3, 1, 1, 3};
  const float alpha_values[] = {0.0f, 1.0f, 2.0f};
  int output_shape[] = {3, 2, 2, 3};
  const float golden[] = {
      0.0f, 0.0f,  0.0f,   // Row 1, Column 1
      1.0f, 1.0f,  1.0f,   // Row 1, Column 2
      0.0f, -1.0f, -2.0f,  // Row 2, Column 1
      0.0f, -2.0f, -4.0f,  // Row 1, Column 2
  };
  const int output_dims_count = 12;
  float output_data[output_dims_count];
  tflite::testing::TestPreluFloat(input_shape, input_values, alpha_shape,
                                  alpha_values, golden, output_shape,
                                  output_data);
}

TF_LITE_MICRO_TEST(QuantizedInt8PreluActivationsOpTest) {
  int input_shape[] = {3, 2, 2, 3};
  const float input_values[] = {
      0.0f,   0.0f,   0.0f,    // Row 1, Column 1
      0.5f,   0.5f,   0.5f,    // Row 1, Column 2
      -1.0f,  -1.0f,  -1.0f,   // Row 2, Column 1
      -0.25f, -0.25f, -0.25f,  // Row 1, Column 2
  };
  int alpha_shape[] = {3, 1, 1, 3};
  const float alpha_values[] = {0.0f, 0.5f, -0.5f};
  int output_shape[] = {3, 2, 2, 3};
  const float golden[] = {
      0.0f, 0.0f,    0.0f,    // Row 1, Column 1
      0.5f, 0.5f,    0.5f,    // Row 1, Column 2
      0.0f, -0.5f,   0.5f,    // Row 2, Column 1
      0.0f, -0.125f, 0.125f,  // Row 1, Column 2
  };
  const int dims_count = 12;
  int8_t input_quantized[dims_count];
  int8_t alpha_quantized[3];
  float scale = 2.0 / 255.0;
  int zero_point = 0;
  int8_t output_data_q[dims_count];
  float output_data_f[dims_count];
  tflite::testing::TestPreluQuantized<int8_t, int8_t>(
      input_shape, input_values, input_quantized, scale, zero_point,
      alpha_shape, alpha_values, alpha_quantized, scale, zero_point, golden,
      scale, zero_point, output_shape, output_data_q, output_data_f);
}

TF_LITE_MICRO_TEST(QuantizedInt16PreluActivationsOpTest) {
  int input_shape[] = {3, 2, 2, 3};
  const float input_values[] = {
      0.0f,   0.0f,   0.0f,    // Row 1, Column 1
      0.5f,   0.5f,   0.5f,    // Row 1, Column 2
      -1.0f,  -1.0f,  -1.0f,   // Row 2, Column 1
      -0.25f, -0.25f, -0.25f,  // Row 1, Column 2
  };
  int alpha_shape[] = {3, 1, 1, 3};
  const float alpha_values[] = {0.0f, 0.5f, -0.5f};
  int output_shape[] = {3, 2, 2, 3};
  const float golden[] = {
      0.0f, 0.0f,    0.0f,    // Row 1, Column 1
      0.5f, 0.5f,    0.5f,    // Row 1, Column 2
      0.0f, -0.5f,   0.5f,    // Row 2, Column 1
      0.0f, -0.125f, 0.125f,  // Row 1, Column 2
  };
  const int dims_count = 12;
  int16_t input_quantized[dims_count];
  int8_t alpha_quantized[3];
  float scale_input_output = 2.0 / 65535.0;
  float scale_alpha = 2.0 / 255.0;
  int zero_point = 0;
  int16_t output_data_q[dims_count];
  float output_data_f[dims_count];
  tflite::testing::TestPreluQuantized<int16_t, int8_t>(
      input_shape, input_values, input_quantized, scale_input_output,
      zero_point, alpha_shape, alpha_values, alpha_quantized, scale_alpha,
      zero_point, golden, scale_input_output, zero_point, output_shape,
      output_data_q, output_data_f);
}
TF_LITE_MICRO_TESTS_END
