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

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

void TestReluFloat(int* input_dims_data, const float* input_data,
                   int* output_dims_data, const float* golden,
                   float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_RELU();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_elements_count; ++i) {
    EXPECT_NEAR(golden[i], output_data[i], 1e-5f);
  }
}

void TestRelu6Float(int* input_dims_data, const float* input_data,
                    int* output_dims_data, const float* golden,
                    float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_RELU6();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_elements_count; ++i) {
    EXPECT_NEAR(golden[i], output_data[i], 1e-5f);
  }
}

void TestReluInt8(int* input_dims_data, const float* input_data,
                  int8_t* input_data_quantized, const float input_scale,
                  const int input_zero_point, const float* golden,
                  int8_t* golden_quantized, int* output_dims_data,
                  const float output_scale, const int output_zero_point,
                  int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_RELU();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  Quantize(golden, golden_quantized, output_elements_count, output_scale,
           output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

void TestReluInt16(int* input_dims_data, const float* input_data,
                   int16_t* input_data_quantized, const float input_scale,
                   const int input_zero_point, const float* golden,
                   int16_t* golden_quantized, int* output_dims_data,
                   const float output_scale, const int output_zero_point,
                   int16_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_RELU();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  Quantize(golden, golden_quantized, output_elements_count, output_scale,
           output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

void TestRelu6Int8(int* input_dims_data, const float* input_data,
                   int8_t* input_data_quantized, const float input_scale,
                   const int input_zero_point, const float* golden,
                   int8_t* golden_quantized, int* output_dims_data,
                   const float output_scale, const int output_zero_point,
                   int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_RELU6();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  Quantize(golden, golden_quantized, output_elements_count, output_scale,
           output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

void TestRelu6Int16(int* input_dims_data, const float* input_data,
                    int16_t* input_data_quantized, const float input_scale,
                    const int input_zero_point, const float* golden,
                    int16_t* golden_quantized, int* output_dims_data,
                    const float output_scale, const int output_zero_point,
                    int16_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_RELU6();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  Quantize(golden, golden_quantized, output_elements_count, output_scale,
           output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(ActivationsTest, SimpleReluTestFloat) {
  const int output_elements_count = 10;
  int input_shape[] = {2, 1, 5};
  const float input_data[] = {
      1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0,
  };
  const float golden[] = {1.0, 2.0, 3.0, 4.0, 5.0, 0, 0, 0, 0, 0};
  int output_shape[] = {2, 1, 5};
  float output_data[output_elements_count];
  tflite::testing::TestReluFloat(input_shape, input_data, output_shape, golden,
                                 output_data);
}

TEST(ActivationsTest, SimpleRelu6TestFloat) {
  const int output_elements_count = 10;
  float output_data[output_elements_count];
  int input_shape[] = {2, 1, 5};
  const float input_data[] = {4.0,  5.0,  6.0,  7.0,  8.0,
                              -4.0, -5.0, -6.0, -7.0, -8.0};
  int output_shape[] = {2, 1, 5};
  const float golden[] = {
      4.0, 5.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  };

  tflite::testing::TestRelu6Float(input_shape, input_data, output_shape, golden,
                                  output_data);
}

TEST(ActivationsTest, SimpleReluTestInt8) {
  const int elements_count = 10;

  int input_shape[] = {2, 1, 5};
  const float input_data[] = {1, 2, 3, 4, 5, -1, -2, -3, -4, -5};
  int8_t input_quantized[elements_count];
  int output_shape[] = {2, 1, 5};
  const float golden[] = {1, 2, 3, 4, 5, 0, 0, 0, 0, 0};
  int8_t golden_quantized[elements_count];
  int8_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 0;
  const float output_scale = 0.5f;
  const int output_zero_point = 0;

  tflite::testing::TestReluInt8(input_shape, input_data, input_quantized,
                                input_scale, input_zero_point, golden,
                                golden_quantized, output_shape, output_scale,
                                output_zero_point, output_data);
}

TEST(ActivationsTest, SimpleReluTestInt16) {
  const int elements_count = 10;

  int input_shape[] = {2, 2, 5};
  const float input_data[] = {256,  257,  258,  259,  260,
                              -256, -257, -258, -259, -260};
  int16_t input_quantized[elements_count];
  int output_shape[] = {2, 2, 5};
  const float golden[] = {256, 257, 258, 259, 260, 0, 0, 0, 0, 0};
  int16_t golden_quantized[elements_count];
  int16_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 0;
  const float output_scale = 0.5f;
  const int output_zero_point = 0;

  tflite::testing::TestReluInt16(input_shape, input_data, input_quantized,
                                 input_scale, input_zero_point, golden,
                                 golden_quantized, output_shape, output_scale,
                                 output_zero_point, output_data);
}

TEST(ActivationsTest, SimpleRelu6TestInt8) {
  const int elements_count = 10;

  int input_shape[] = {2, 1, 5};
  const float input_data[] = {4, 5, 6, 7, 8, -1, -2, -3, -4, -5};
  int8_t input_quantized[elements_count];
  int output_shape[] = {2, 1, 5};
  const float golden[] = {4, 5, 6, 6, 6, 0, 0, 0, 0, 0};
  int8_t golden_quantized[elements_count];
  int8_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 127;
  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  tflite::testing::TestRelu6Int8(input_shape, input_data, input_quantized,
                                 input_scale, input_zero_point, golden,
                                 golden_quantized, output_shape, output_scale,
                                 output_zero_point, output_data);
}

TEST(ActivationsTest, SimpleRelu6TestInt16) {
  const int elements_count = 10;

  int input_shape[] = {2, 1, 5};
  const float input_data[] = {4, 5, 6, 7, 8, -1, -2, -3, -4, -5};
  int16_t input_quantized[elements_count];
  int output_shape[] = {2, 1, 5};
  const float golden[] = {4, 5, 6, 6, 6, 0, 0, 0, 0, 0};
  int16_t golden_quantized[elements_count];
  int16_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 0;
  const float output_scale = 0.5f;
  const int output_zero_point = 0;

  tflite::testing::TestRelu6Int16(input_shape, input_data, input_quantized,
                                  input_scale, input_zero_point, golden,
                                  golden_quantized, output_shape, output_scale,
                                  output_zero_point, output_data);
}

TEST(ActivationsTest, ReluMultiElementFloat) {
  constexpr int kElementsCount = 21;
  int input_shape[] = {2, 3, 7};
  const float input_data[kElementsCount] = {
      -10.0f, -5.5f, -1.0f, -0.25f, 0.0f,  0.25f, 1.0f,
      2.5f,   5.0f,  10.0f, -3.0f,  4.0f,  -0.5f, 0.5f,
      -8.0f,  8.0f,  -2.0f, 2.0f,   -0.1f, 0.1f,  7.5f,
  };
  const float golden[kElementsCount] = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.25f, 1.0f, 2.5f, 5.0f, 10.0f, 0.0f,
      4.0f, 0.0f, 0.5f, 0.0f, 8.0f, 0.0f,  2.0f, 0.0f, 0.1f, 7.5f,
  };
  int output_shape[] = {2, 3, 7};
  float output_data[kElementsCount];
  tflite::testing::TestReluFloat(input_shape, input_data, output_shape, golden,
                                 output_data);
}

TEST(ActivationsTest, ReluMultiElementInt8Asymmetric) {
  constexpr int kElementsCount = 21;
  int input_shape[] = {2, 3, 7};
  const float input_data[kElementsCount] = {
      -8.0f, -4.0f, -1.5f, -0.25f, 0.0f, 0.25f, 0.5f, 1.0f,  2.0f,  3.5f,  5.0f,
      7.5f,  10.0f, 12.0f, -6.0f,  6.0f, -2.5f, 2.5f, -0.5f, 1.25f, 15.0f,
  };
  const float golden[kElementsCount] = {
      0.0f, 0.0f,  0.0f,  0.0f, 0.0f, 0.25f, 0.5f, 1.0f, 2.0f,  3.5f,  5.0f,
      7.5f, 10.0f, 12.0f, 0.0f, 6.0f, 0.0f,  2.5f, 0.0f, 1.25f, 15.0f,
  };
  int8_t input_quantized[kElementsCount];
  int output_shape[] = {2, 3, 7};
  int8_t golden_quantized[kElementsCount];
  int8_t output_data[kElementsCount];

  const float input_scale = 0.25f;
  const int input_zero_point = -10;
  const float output_scale = 0.125f;
  const int output_zero_point = -128;

  tflite::testing::TestReluInt8(input_shape, input_data, input_quantized,
                                input_scale, input_zero_point, golden,
                                golden_quantized, output_shape, output_scale,
                                output_zero_point, output_data);
}

TEST(ActivationsTest, Relu6MultiElementFloat) {
  constexpr int kElementsCount = 21;
  int input_shape[] = {2, 3, 7};
  const float input_data[kElementsCount] = {
      -8.0f, -3.5f, -0.1f, 0.0f, 0.5f, 1.5f,  3.0f, 4.5f, 5.9f, 6.0f, 6.1f,
      8.0f,  12.0f, -1.0f, 2.0f, 7.0f, -5.0f, 5.0f, 0.0f, 6.0f, 9.5f,
  };
  const float golden[kElementsCount] = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 1.5f, 3.0f, 4.5f, 5.9f, 6.0f, 6.0f,
      6.0f, 6.0f, 0.0f, 2.0f, 6.0f, 0.0f, 5.0f, 0.0f, 6.0f, 6.0f,
  };
  int output_shape[] = {2, 3, 7};
  float output_data[kElementsCount];
  tflite::testing::TestRelu6Float(input_shape, input_data, output_shape, golden,
                                  output_data);
}

TEST(ActivationsTest, Relu6MultiElementInt8) {
  constexpr int kElementsCount = 21;
  int input_shape[] = {2, 3, 7};
  const float input_data[kElementsCount] = {
      -8.0f, -3.5f, -0.5f, 0.0f, 0.5f, 1.5f,  3.0f, 4.5f, 5.5f, 6.0f, 6.5f,
      8.0f,  10.0f, -1.0f, 2.0f, 7.0f, -5.0f, 5.0f, 0.0f, 6.0f, 9.5f,
  };
  const float golden[kElementsCount] = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 1.5f, 3.0f, 4.5f, 5.5f, 6.0f, 6.0f,
      6.0f, 6.0f, 0.0f, 2.0f, 6.0f, 0.0f, 5.0f, 0.0f, 6.0f, 6.0f,
  };
  int8_t input_quantized[kElementsCount];
  int output_shape[] = {2, 3, 7};
  int8_t golden_quantized[kElementsCount];
  int8_t output_data[kElementsCount];

  const float input_scale = 0.125f;
  const int input_zero_point = -20;
  const float output_scale = 0.125f;
  const int output_zero_point = -20;

  tflite::testing::TestRelu6Int8(input_shape, input_data, input_quantized,
                                 input_scale, input_zero_point, golden,
                                 golden_quantized, output_shape, output_scale,
                                 output_zero_point, output_data);
}

TF_LITE_MICRO_TESTS_MAIN
