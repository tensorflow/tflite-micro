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

#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {

void TestElementwiseFloat(const TFLMRegistration& registration,
                          int* input_dims_data, const float* input_data,
                          int* output_dims_data,
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

  static int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  static int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

template <typename T>
void TestElementwiseQuantized(const TFLMRegistration& registration,
                              int* input_dims_data, const float* input_data,
                              T* input_quantized, float input_scale,
                              int32_t input_zero_point, int* output_dims_data,
                              const float* expected_output_data, T* output_data,
                              const float output_scale,
                              const int output_zero_point,
                              TfLiteStatus expected_invoke_status = kTfLiteOk) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;

  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point)};

  int input_zero_points[2] = {1, input_zero_point};
  float input_scales[2] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
  tensors[0].quantization = {kTfLiteAffineQuantization, &input_quant};

  int output_zero_points[2] = {1, output_zero_point};
  float output_scales[2] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points), 0};
  tensors[1].quantization = {kTfLiteAffineQuantization, &output_quant};

  static int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  static int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(expected_invoke_status, runner.Invoke());

  if (expected_invoke_status == kTfLiteOk) {
    for (int i = 0; i < output_dims_count; ++i) {
      float f = (output_data[i] - output_zero_point) * output_scale;
      EXPECT_NEAR(expected_output_data[i], f, input_scale);
    }
  }
}

void TestElementwiseBool(const TFLMRegistration& registration,
                         int* input_dims_data, const bool* input_data,
                         int* output_dims_data,
                         const bool* expected_output_data, bool* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input_data, input_dims),
                                        CreateTensor(output_data, output_dims)};

  // Place false in the uninitialized output buffer.
  for (int i = 0; i < output_dims_count; ++i) {
    output_data[i] = false;
  }

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace testing
}  // namespace tflite

TEST(ElementwiseTest, Abs) {
  constexpr int output_dims_count = 4;
  int shape[] = {2, 2, 2};
  const float input[] = {0.01, -0.01, 10, -10};
  const float golden[] = {0.01, 0.01, 10, 10};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_ABS(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, AbsInt8) {
  int shape[] = {2, 1, 8};

  const float input_data[] = {15., 46., 78., -142., -1., -17., -49., 113.};
  int8_t input_quantized[8];

  const float golden[] = {15., 46., 78., 142., 1., 17., 49., 113.};
  int8_t output_quantized[8];

  const float abs_max = 142;
  const float data_min = -142;
  const float data_max = 113;
  const float input_scale = (data_max - data_min) / 255.0f;
  const float output_scale = abs_max / 255.0f;
  const int input_zero_point = 127 - data_max;
  const int output_zero_point = -128;
  tflite::testing::TestElementwiseQuantized<int8_t>(
      tflite::Register_ABS(), shape, input_data, input_quantized, input_scale,
      input_zero_point, shape, golden, output_quantized, output_scale,
      output_zero_point);
}

TEST(ElementwiseTest, AbsInt8SameScale) {
  int shape[] = {2, 1, 8};

  const float input_data[] = {15., 46., 78., -142., -1., -17., -49., 113.};
  int8_t input_quantized[8];

  const float golden[] = {15., 46., 78., 142., 1., 17., 49., 113.};
  int8_t output_quantized[8];

  const float data_min = -142;
  const float data_max = 113;
  const float scale = (data_max - data_min) / 255.0f;
  const int zero_point = 127 - data_max;
  tflite::testing::TestElementwiseQuantized(
      tflite::Register_ABS(), shape, input_data, input_quantized, scale,
      zero_point, shape, golden, output_quantized, scale, -128);
}

TEST(ElementwiseTest, AbsInt16) {
  int shape[] = {2, 1, 8};

  const float input_data[] = {15., 46., 78., -142., -1., -17., -49., 113.};
  int16_t input_quantized[8];

  const float golden[] = {15., 46., 78., 142., 1., 17., 49., 113.};
  int16_t output_quantized[8];

  const float input_max = 142;
  const float output_max = 150;
  const float input_scale = input_max / std::numeric_limits<int16_t>::max();
  const float output_scale = output_max / std::numeric_limits<int16_t>::max();
  tflite::testing::TestElementwiseQuantized(
      tflite::Register_ABS(), shape, input_data, input_quantized, input_scale,
      /*input_zero_point*/ 0, shape, golden, output_quantized, output_scale,
      /*output_zero_point*/ 0);
}

TEST(ElementwiseTest, Sin) {
  constexpr int output_dims_count = 4;
  int shape[] = {2, 2, 2};
  const float input[] = {0, 3.1415926, -3.1415926, 1};
  const float golden[] = {0, 0, 0, 0.84147};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_SIN(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, Cos) {
  constexpr int output_dims_count = 4;
  int shape[] = {2, 2, 2};
  const float input[] = {0, 3.1415926, -3.1415926, 1};
  const float golden[] = {1, -1, -1, 0.54030};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_COS(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, Log) {
  constexpr int output_dims_count = 4;
  int shape[] = {2, 2, 2};
  const float input[] = {1, 2.7182818, 0.5, 2};
  const float golden[] = {0, 1, -0.6931472, 0.6931472};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_LOG(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, Sqrt) {
  constexpr int output_dims_count = 4;
  int shape[] = {2, 2, 2};
  const float input[] = {0, 1, 2, 4};
  const float golden[] = {0, 1, 1.41421, 2};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_SQRT(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, Rsqrt) {
  constexpr int output_dims_count = 4;
  int shape[] = {2, 2, 2};
  const float input[] = {1, 2, 4, 9};
  const float golden[] = {1, 0.7071, 0.5, 0.33333};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_RSQRT(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, RsqrtInt8) {
  int shape[] = {2, 1, 8};

  const float input_data[] = {15., 46., 78., 142., 1., 17., 49., 113.};
  int8_t input_quantized[8];

  const float golden[] = {0.2582, 0.14744, 0.11323,  0.08392,
                          1.,     0.24254, 0.142857, 0.09407};
  int8_t output_quantized[8];

  const float data_max = 142;
  const float input_scale = 142.0 / 255.0;
  const float output_scale = 1.0 / 255.0;
  const int input_zero_point = 127 - data_max;
  const int output_zero_point = -128;
  tflite::testing::TestElementwiseQuantized<int8_t>(
      tflite::Register_RSQRT(), shape, input_data, input_quantized, input_scale,
      input_zero_point, shape, golden, output_quantized, output_scale,
      output_zero_point);
}

TEST(ElementwiseTest, RsqrtInt16) {
  int shape[] = {2, 1, 8};

  const float input_data[] = {15., 46., 78., 142., 1., 17., 49., 113.};
  int16_t input_quantized[8];

  const float golden[] = {0.2582, 0.14744, 0.11323,  0.08392,
                          1.,     0.24254, 0.142857, 0.09407};
  int16_t output_quantized[8];

  const float input_scale = 142.0 / 32768.0;
  const float output_scale = 1.0 / 32768.0;
  const int input_zero_point = 0;
  const int output_zero_point = 0;
  tflite::testing::TestElementwiseQuantized<int16_t>(
      tflite::Register_RSQRT(), shape, input_data, input_quantized, input_scale,
      input_zero_point, shape, golden, output_quantized, output_scale,
      output_zero_point);
}

TEST(ElementwiseTest, RsqrtCloseTo0Int8) {
  int shape[] = {2, 1, 8};

  const float input_data[] = {15., 46., 78., 142., 1., 0.1, 49., 113.};
  int8_t input_quantized[8];

  const float golden[] = {0.2582, 0.14744, 0.11323,  0.08392,
                          1.,     3.16228, 0.142857, 0.09407};
  int8_t output_quantized[8];

  const float data_max = 142;
  const float input_scale = 142.0 / 255.0;
  const float output_scale = 3.16 / 255.0;
  const int input_zero_point = 127 - data_max;
  const int output_zero_point = -128;
  tflite::testing::TestElementwiseQuantized<int8_t>(
      tflite::Register_RSQRT(), shape, input_data, input_quantized, input_scale,
      input_zero_point, shape, golden, output_quantized, output_scale,
      output_zero_point);
}

TEST(ElementwiseTest, RsqrtNanInt8) {
  int shape[] = {2, 1, 8};

  const float input_data[] = {15., 46., 78., 142., 1., 17., -49., 113.};
  int8_t input_quantized[8];

  const float golden[] = {0.2582, 0.14744, 0.11323,  0.08392,
                          1.,     0.24254, 0.142857, 0.09407};
  int8_t output_quantized[8];

  const float data_max = 142;
  const float input_scale = 142.0 / 255.0;
  const float output_scale = 1.0 / 255.0;
  const int input_zero_point = 127 - data_max;
  const int output_zero_point = -128;

  tflite::testing::TestElementwiseQuantized<int8_t>(
      tflite::Register_RSQRT(), shape, input_data, input_quantized, input_scale,
      input_zero_point, shape, golden, output_quantized, output_scale,
      output_zero_point, kTfLiteError);
}

TEST(ElementwiseTest, Square) {
  constexpr int output_dims_count = 4;
  int shape[] = {2, 2, 2};
  const float input[] = {1, 2, 0.5, -3.0};
  const float golden[] = {1, 4.0, 0.25, 9.0};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_SQUARE(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, LogicalNot) {
  constexpr int output_dims_count = 4;
  int shape[] = {2, 2, 2};
  const bool input[] = {true, false, false, true};
  const bool golden[] = {false, true, true, false};
  bool output_data[output_dims_count];
  tflite::testing::TestElementwiseBool(tflite::Register_LOGICAL_NOT(), shape,
                                       input, shape, golden, output_data);
}

TEST(ElementwiseTest, AbsMultiElement) {
  constexpr int output_dims_count = 12;
  int shape[] = {2, 3, 4};
  const float input[] = {0.01f, -0.01f, 10.0f,  -10.0f,  0.0f,   -1.5f,
                         2.5f,  -3.75f, 100.0f, -100.0f, 0.125f, -0.125f};
  const float golden[] = {0.01f, 0.01f, 10.0f,  10.0f,  0.0f,   1.5f,
                          2.5f,  3.75f, 100.0f, 100.0f, 0.125f, 0.125f};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_ABS(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, SinMultiElement) {
  constexpr int output_dims_count = 12;
  int shape[] = {2, 3, 4};
  const float input[] = {0.0f,  3.1415926f, -3.1415926f, 1.0f,  -1.0f,  0.5f,
                         -0.5f, 1.5707963f, -1.5707963f, 0.25f, -0.25f, 2.0f};
  const float golden[] = {0.0f,        0.0f,       0.0f,        0.8414709f,
                          -0.8414709f, 0.4794255f, -0.4794255f, 1.0f,
                          -1.0f,       0.2474039f, -0.2474039f, 0.9092974f};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_SIN(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, CosMultiElement) {
  constexpr int output_dims_count = 12;
  int shape[] = {2, 3, 4};
  const float input[] = {0.0f,  3.1415926f, -3.1415926f, 1.0f,  -1.0f,  0.5f,
                         -0.5f, 1.5707963f, -1.5707963f, 0.25f, -0.25f, 2.0f};
  const float golden[] = {1.0f,       -1.0f,      -1.0f,      0.5403023f,
                          0.5403023f, 0.8775826f, 0.8775826f, 0.0f,
                          0.0f,       0.9689124f, 0.9689124f, -0.4161468f};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_COS(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, LogMultiElement) {
  constexpr int output_dims_count = 12;
  int shape[] = {2, 3, 4};
  const float input[] = {1.0f, 2.7182818f, 0.5f,  2.0f, 4.0f,  0.25f,
                         8.0f, 0.125f,     10.0f, 0.1f, 16.0f, 0.0625f};
  const float golden[] = {0.0f,       1.0f,        -0.6931472f, 0.6931472f,
                          1.3862944f, -1.3862944f, 2.0794415f,  -2.0794415f,
                          2.3025851f, -2.3025851f, 2.7725887f,  -2.7725887f};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_LOG(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, SqrtMultiElement) {
  constexpr int output_dims_count = 12;
  int shape[] = {2, 3, 4};
  const float input[] = {0.0f,  1.0f,  2.0f,  4.0f,    9.0f,  16.0f,
                         25.0f, 0.25f, 0.01f, 0.0625f, 36.0f, 49.0f};
  const float golden[] = {0.0f, 1.0f, 1.4142135f, 2.0f,  3.0f, 4.0f,
                          5.0f, 0.5f, 0.1f,       0.25f, 6.0f, 7.0f};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_SQRT(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, RsqrtMultiElement) {
  constexpr int output_dims_count = 12;
  int shape[] = {2, 3, 4};
  const float input[] = {1.0f,  2.0f,  4.0f,   9.0f,    16.0f, 25.0f,
                         0.25f, 0.04f, 100.0f, 0.0625f, 64.0f, 400.0f};
  const float golden[] = {1.0f, 0.7071068f, 0.5f, 0.3333333f, 0.25f,  0.2f,
                          2.0f, 5.0f,       0.1f, 4.0f,       0.125f, 0.05f};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_RSQRT(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, SquareMultiElement) {
  constexpr int output_dims_count = 12;
  int shape[] = {2, 3, 4};
  const float input[] = {1.0f, 2.0f,  0.5f, -3.0f, 0.0f,  -0.25f,
                         4.0f, -5.0f, 1.5f, -2.5f, 10.0f, -0.1f};
  const float golden[] = {1.0f,  4.0f,  0.25f, 9.0f,  0.0f,   0.0625f,
                          16.0f, 25.0f, 2.25f, 6.25f, 100.0f, 0.01f};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::Register_SQUARE(), shape, input,
                                        shape, golden, output_data);
}

TEST(ElementwiseTest, LogicalNotMultiElement) {
  constexpr int output_dims_count = 26;
  int shape[] = {2, 2, 13};
  const bool input[] = {true,  false, false, true,  true,  false, true,
                        false, false, true,  false, true,  true,  false,
                        false, false, true,  true,  false, true,  false,
                        false, true,  false, true,  false};
  const bool golden[] = {false, true,  true,  false, false, true,  false,
                         true,  true,  false, true,  false, false, true,
                         true,  true,  false, false, true,  false, true,
                         true,  false, true,  false, true};
  bool output_data[output_dims_count];
  tflite::testing::TestElementwiseBool(tflite::Register_LOGICAL_NOT(), shape,
                                       input, shape, golden, output_data);
}

TF_LITE_MICRO_TESTS_MAIN
