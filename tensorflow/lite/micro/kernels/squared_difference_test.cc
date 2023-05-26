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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kNumTestShapes = 4;
constexpr int kMaxTestShapeSize = 5;

int test_shape[kNumTestShapes][kMaxTestShapeSize] = {
    {1, 6},
    {2, 2, 3},
    {3, 2, 1, 3},
    {4, 1, 3, 1, 2},
};

template <typename T>
void ValidateSquaredDifferenceGoldens(TfLiteTensor* tensors, int tensors_size,
                                      const T* golden, T* output,
                                      int output_size, float tolerance = 1e-5) {
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_SQUARED_DIFFERENCE();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], tolerance);
  }
}

template <typename T>
void TestSquaredDifference(int* input1_dims_data, const T* input1_data,
                           int* input2_dims_data, const T* input2_data,
                           int* output_dims_data, const T* expected_output,
                           T* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  ValidateSquaredDifferenceGoldens(tensors, tensors_size, expected_output,
                                   output_data, ElementCount(*output_dims));
}

template <typename T>
void TestSquaredDifferenceQuantized(
    int* input1_dims_data, const float* input1_data, T* input1_quantized,
    float input1_min, float input1_max,

    int* input2_dims_data, const float* input2_data, T* input2_quantized,
    float input2_min, float input2_max,

    int* output_dims_data, T* output_data, float output_min, float output_max,
    float* dequantized_output, const float* golden,

    float tolerance, bool narrow_range = false) {
  QuantizationParams input1_qparams;
  QuantizationParams input2_qparams;
  QuantizationParams output_qparams;

  input1_qparams = ChooseQuantizationParams<T>(static_cast<double>(input1_min),
                                               static_cast<double>(input1_max),
                                               narrow_range);
  input2_qparams = ChooseQuantizationParams<T>(static_cast<double>(input2_min),
                                               static_cast<double>(input2_max),
                                               narrow_range);
  output_qparams = ChooseQuantizationParams<T>(static_cast<double>(output_min),
                                               static_cast<double>(output_max),
                                               narrow_range);

  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  int output_size = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor<T>(input1_data, input1_quantized, input1_dims,
                               input1_qparams.scale, input1_qparams.zero_point),
      CreateQuantizedTensor<T>(input2_data, input2_quantized, input2_dims,
                               input2_qparams.scale, input2_qparams.zero_point),
      CreateQuantizedTensor<T>(output_data, output_dims, output_qparams.scale,
                               output_qparams.zero_point),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_SQUARED_DIFFERENCE();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  Dequantize(output_data, output_size, output_qparams.scale,
             output_qparams.zero_point, dequantized_output);

  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], dequantized_output[i], tolerance);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatSquaredDifferenceSameShape) {
  constexpr int data_size = 4;
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-0.2, 0.2, -1.2, 0.8};
  const float input2_values[] = {0.5, 0.2, -1.5, 0.5};
  const float golden_values[] = {0.49, 0.0, 0.09, 0.09};
  float output_data[data_size];
  tflite::testing::TestSquaredDifference(
      inout_shape, input1_values, inout_shape, input2_values, inout_shape,
      golden_values, output_data);
}

TF_LITE_MICRO_TEST(FloatSquaredDifferenceVariousShapes) {
  constexpr int data_size = 6;
  const float input1_values[] = {-2.0, 0.2, 0.3, 0.8, 1.1, -2.0};
  const float input2_values[] = {1.0, 0.2, 0.6, 0.4, -1.0, -0.0};
  const float golden_values[] = {9.0, 0.0, 0.09, 0.16, 4.41, 4.0};
  float output_data[data_size];
  for (int i = 0; i < tflite::testing::kNumTestShapes; ++i) {
    tflite::testing::TestSquaredDifference(
        tflite::testing::test_shape[i], input1_values,
        tflite::testing::test_shape[i], input2_values,
        tflite::testing::test_shape[i], golden_values, output_data);
  }
}

TF_LITE_MICRO_TEST(FloatSquaredDifferenceWithBroadcast) {
  constexpr int data_size = 6;

  // input 2 is scalar
  int input2_shape[] = {1, 1};
  const float input1_values[] = {-0.2, 0.2, 0.5, 0.8, 0.11, 1.1};
  const float input2_values[] = {0.1};
  const float golden_values[] = {0.09, 0.01, 0.16, 0.49, 0.0001, 1.0};
  float output_data[data_size];
  for (int i = 0; i < tflite::testing::kNumTestShapes; ++i) {
    tflite::testing::TestSquaredDifference(
        tflite::testing::test_shape[i], input1_values, input2_shape,
        input2_values, tflite::testing::test_shape[i], golden_values,
        output_data);
  }
}

TF_LITE_MICRO_TEST(IntegerSquaredDifferenceSameShape) {
  constexpr int data_size = 4;
  int inout_shape[] = {4, 1, 2, 2, 1};
  const int32_t input1_values[] = {-2, 2, -15, 8};
  const int32_t input2_values[] = {5, -2, -3, 5};
  const int32_t golden_values[] = {49, 16, 144, 9};
  int32_t output_data[data_size];
  tflite::testing::TestSquaredDifference(
      inout_shape, input1_values, inout_shape, input2_values, inout_shape,
      golden_values, output_data);
}

TF_LITE_MICRO_TEST(IntegerSquaredDifferenceVariousShapes) {
  constexpr int data_size = 6;
  const int32_t input1_values[] = {-20, 2, 3, 8, 11, -20};
  const int32_t input2_values[] = {1, 2, 6, 5, -5, -20};
  const int32_t golden_values[] = {441, 0, 9, 9, 256, 0};
  int32_t output_data[data_size];
  for (int i = 0; i < tflite::testing::kNumTestShapes; ++i) {
    tflite::testing::TestSquaredDifference(
        tflite::testing::test_shape[i], input1_values,
        tflite::testing::test_shape[i], input2_values,
        tflite::testing::test_shape[i], golden_values, output_data);
  }
}

TF_LITE_MICRO_TEST(IntegerSquaredDifferenceWithBroadcast) {
  constexpr int data_size = 6;

  // input 2 is a scalar
  int input2_shape[] = {1, 1};
  const int32_t input1_values[] = {-20, 10, 7, 3, 1, 13};
  const int32_t input2_values[] = {3};
  const int32_t golden_values[] = {529, 49, 16, 0, 4, 100};
  int32_t output_data[data_size];
  for (int i = 0; i < tflite::testing::kNumTestShapes; ++i) {
    tflite::testing::TestSquaredDifference(
        tflite::testing::test_shape[i], input1_values, input2_shape,
        input2_values, tflite::testing::test_shape[i], golden_values,
        output_data);
  }
}

TF_LITE_MICRO_TEST(QuantizedSquaredDifferenceSameShape) {
  constexpr int data_size = 4;
  int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-0.2, 0.2, -1.2, 0.8};
  const float input2_values[] = {0.5, 0.2, -1.5, 0.5};
  const float golden_values[] = {0.49, 0.0, 0.09, 0.09};
  float output_dequantized[data_size];
  // Int8 case
  int8_t input1_int8[data_size];
  int8_t input2_int8[data_size];
  int8_t output_int8[data_size];
  tflite::testing::TestSquaredDifferenceQuantized(
      inout_shape, input1_values, input1_int8, -1.2f, 0.8f, inout_shape,
      input2_values, input2_int8, -1.5f, 0.5f, inout_shape, output_int8, 0.0f,
      0.5f, output_dequantized, golden_values, 2.0f / 255.0f);

  // Int16 case
  int16_t input1_int16[data_size];
  int16_t input2_int16[data_size];
  int16_t output_int16[data_size];
  // Symmetrical quantization: (rmin == -rmax), requires narrow range (qmin =
  // -qmax).
  // TODO(b/269352046): understand the tolerance level
  // http://b/269352046#comment7
  tflite::testing::TestSquaredDifferenceQuantized(
      inout_shape, input1_values, input1_int16, -1.2f, 1.2f, inout_shape,
      input2_values, input2_int16, -1.5f, 1.5f, inout_shape, output_int16,
      -0.5f, 0.5f, output_dequantized, golden_values, 6.0f / 32768.0f,
      /*narrow_range=*/true);
}

TF_LITE_MICRO_TEST(QuantizedSquaredDifferenceVariousShapes) {
  constexpr int data_size = 6;
  const float input1_values[] = {-2.0, 0.2, 0.3, 0.8, 1.1, -2.0};
  const float input2_values[] = {1.0, 0.2, 0.6, 0.4, -1.0, -0.0};
  const float golden_values[] = {9.0, 0.0, 0.09, 0.16, 4.41, 4.0};
  // Int8 case
  int8_t input1_int8[data_size];
  int8_t input2_int8[data_size];
  int8_t output_int8[data_size];
  float output_dequantized[data_size];
  for (int i = 0; i < tflite::testing::kNumTestShapes; ++i) {
    tflite::testing::TestSquaredDifferenceQuantized(
        tflite::testing::test_shape[i], input1_values, input1_int8, -2.0f, 1.7f,
        tflite::testing::test_shape[i], input2_values, input2_int8, -1.0f, 1.0f,
        tflite::testing::test_shape[i], output_int8, 0.0f, 9.0f,
        output_dequantized, golden_values, 18.0f / 255.0f);
  }

  // Int16 case
  int16_t input1_int16[data_size];
  int16_t input2_int16[data_size];
  int16_t output_int16[data_size];
  // Symmetrical quantization: (rmin == -rmax), requires narrow range (qmin =
  // -qmax).
  for (int i = 0; i < tflite::testing::kNumTestShapes; ++i) {
    tflite::testing::TestSquaredDifferenceQuantized(
        tflite::testing::test_shape[i], input1_values, input1_int16, -2.0f,
        2.0f, tflite::testing::test_shape[i], input2_values, input2_int16,
        -1.0f, 1.0f, tflite::testing::test_shape[i], output_int16, -9.0f, 9.0f,
        output_dequantized, golden_values, 18.0f / 32768.0f,
        /*narrow_range=*/true);
  }
}

TF_LITE_MICRO_TEST(FloatSquaredDifferenceWithBroadcast) {
  constexpr int data_size = 6;

  // input 2 is a scalar
  int input2_shape[] = {1, 1};
  const float input1_values[] = {-0.2, 0.2, 0.5, 0.8, 0.11, 1.1};
  const float input2_values[] = {0.1};
  const float golden_values[] = {0.09, 0.01, 0.16, 0.49, 0.0001, 1.0};

  // Int8 case
  int8_t input1_int8[data_size];
  int8_t input2_int8[data_size];
  int8_t output_int8[data_size];
  float output_dequantized[data_size];
  for (int i = 0; i < tflite::testing::kNumTestShapes; ++i) {
    tflite::testing::TestSquaredDifferenceQuantized(
        tflite::testing::test_shape[i], input1_values, input1_int8, -0.2f, 1.1f,
        input2_shape, input2_values, input2_int8, 0.0f, 1.0f,
        tflite::testing::test_shape[i], output_int8, 0.0f, 1.0f,
        output_dequantized, golden_values, 2.0f / 255.0f);
  }

  // Int16 case
  int16_t input1_int16[data_size];
  int16_t input2_int16[data_size];
  int16_t output_int16[data_size];
  for (int i = 0; i < tflite::testing::kNumTestShapes; ++i) {
    tflite::testing::TestSquaredDifferenceQuantized(
        tflite::testing::test_shape[i], input1_values, input1_int16, -1.1f,
        1.1f, input2_shape, input2_values, input2_int16, -1.0f, 1.0f,
        tflite::testing::test_shape[i], output_int16, -1.0f, 1.0f,
        output_dequantized, golden_values, 2.0f / 32768.0f,
        /*narrow_range=*/true);
  }
}

TF_LITE_MICRO_TESTS_END
