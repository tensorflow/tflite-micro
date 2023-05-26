/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// The Logistic kernel assumes an output in the range [0, 1.0], leading to these
// quantization parameters.
const float quantized_output_scale_int8 = 1.0 / 255.0;
const int quantized_output_zero_point_int8 = -128;

const int flat_size_basic = 10;
int shape_basic[] = {2, 2, 5};
const float input_data_basic[] = {1, 2, 3, 4, 5, -1, -2, -3, -4, -5};
const float golden_basic[] = {0.73105858, 0.88079708, 0.95257413, 0.98201379,
                              0.99330715, 0.26894142, 0.11920292, 0.04742587,
                              0.01798621, 0.00669285};

const int flat_size_wide_range = 10;
int shape_wide_range[] = {2, 1, 5};
const float input_data_wide_range[]{
    1.0, 2.0, 3.0, 4.0, 93.0, -1.0, -2.0, -3.0, -4.0, -93.0,
};
const float golden_wide_range[] = {
    0.73105858, 0.88079708, 0.95257413, 0.98201379, 1.0,
    0.26894142, 0.11920292, 0.04742587, 0.01798621, 0.0,
};

// Test vector and expected results are directly ported from TensorFlow Lite's
// int16 logistic test.
constexpr int int16_vec_size = 177;

int shape_int16_vec[] = {2, 1, int16_vec_size};

const float int16_input_vec_fp[int16_vec_size] = {
    -20.0000000000, -19.7727272727, -19.5454545455, -19.3181818182,
    -19.0909090909, -18.8636363636, -18.6363636364, -18.4090909091,
    -18.1818181818, -17.9545454545, -17.7272727273, -17.5000000000,
    -17.2727272727, -17.0454545455, -16.8181818182, -16.5909090909,
    -16.3636363636, -16.1363636364, -15.9090909091, -15.6818181818,
    -15.4545454545, -15.2272727273, -15.0000000000, -14.7727272727,
    -14.5454545455, -14.3181818182, -14.0909090909, -13.8636363636,
    -13.6363636364, -13.4090909091, -13.1818181818, -12.9545454545,
    -12.7272727273, -12.5000000000, -12.2727272727, -12.0454545455,
    -11.8181818182, -11.5909090909, -11.3636363636, -11.1363636364,
    -10.9090909091, -10.6818181818, -10.4545454545, -10.2272727273,
    -10.0000000000, -9.7727272727,  -9.5454545455,  -9.3181818182,
    -9.0909090909,  -8.8636363636,  -8.6363636364,  -8.4090909091,
    -8.1818181818,  -7.9545454545,  -7.7272727273,  -7.5000000000,
    -7.2727272727,  -7.0454545455,  -6.8181818182,  -6.5909090909,
    -6.3636363636,  -6.1363636364,  -5.9090909091,  -5.6818181818,
    -5.4545454545,  -5.2272727273,  -5.0000000000,  -4.7727272727,
    -4.5454545455,  -4.3181818182,  -4.0909090909,  -3.8636363636,
    -3.6363636364,  -3.4090909091,  -3.1818181818,  -2.9545454545,
    -2.7272727273,  -2.5000000000,  -2.2727272727,  -2.0454545455,
    -1.8181818182,  -1.5909090909,  -1.3636363636,  -1.1363636364,
    -0.9090909091,  -0.6818181818,  -0.4545454545,  -0.2272727273,
    0.0000000000,   0.2272727273,   0.4545454545,   0.6818181818,
    0.9090909091,   1.1363636364,   1.3636363636,   1.5909090909,
    1.8181818182,   2.0454545455,   2.2727272727,   2.5000000000,
    2.7272727273,   2.9545454545,   3.1818181818,   3.4090909091,
    3.6363636364,   3.8636363636,   4.0909090909,   4.3181818182,
    4.5454545455,   4.7727272727,   5.0000000000,   5.2272727273,
    5.4545454545,   5.6818181818,   5.9090909091,   6.1363636364,
    6.3636363636,   6.5909090909,   6.8181818182,   7.0454545455,
    7.2727272727,   7.5000000000,   7.7272727273,   7.9545454545,
    8.1818181818,   8.4090909091,   8.6363636364,   8.8636363636,
    9.0909090909,   9.3181818182,   9.5454545455,   9.7727272727,
    10.0000000000,  10.2272727273,  10.4545454545,  10.6818181818,
    10.9090909091,  11.1363636364,  11.3636363636,  11.5909090909,
    11.8181818182,  12.0454545455,  12.2727272727,  12.5000000000,
    12.7272727273,  12.9545454545,  13.1818181818,  13.4090909091,
    13.6363636364,  13.8636363636,  14.0909090909,  14.3181818182,
    14.5454545455,  14.7727272727,  15.0000000000,  15.2272727273,
    15.4545454545,  15.6818181818,  15.9090909091,  16.1363636364,
    16.3636363636,  16.5909090909,  16.8181818182,  17.0454545455,
    17.2727272727,  17.5000000000,  17.7272727273,  17.9545454545,
    18.1818181818,  18.4090909091,  18.6363636364,  18.8636363636,
    19.0909090909,  19.3181818182,  19.5454545455,  19.7727272727,
    20.0000000000};

const float int16_golden_vec_fp[int16_vec_size] = {
    0.0000000021, 0.0000000026, 0.0000000032, 0.0000000041, 0.0000000051,
    0.0000000064, 0.0000000081, 0.0000000101, 0.0000000127, 0.0000000159,
    0.0000000200, 0.0000000251, 0.0000000315, 0.0000000396, 0.0000000497,
    0.0000000623, 0.0000000782, 0.0000000982, 0.0000001232, 0.0000001547,
    0.0000001942, 0.0000002437, 0.0000003059, 0.0000003840, 0.0000004819,
    0.0000006049, 0.0000007593, 0.0000009530, 0.0000011962, 0.0000015014,
    0.0000018846, 0.0000023654, 0.0000029690, 0.0000037266, 0.0000046776,
    0.0000058711, 0.0000073693, 0.0000092497, 0.0000116100, 0.0000145724,
    0.0000182909, 0.0000229581, 0.0000288162, 0.0000361690, 0.0000453979,
    0.0000569815, 0.0000715205, 0.0000897689, 0.0001126729, 0.0001414198,
    0.0001774998, 0.0002227827, 0.0002796147, 0.0003509396, 0.0004404502,
    0.0005527786, 0.0006937345, 0.0008706021, 0.0010925128, 0.0013709094,
    0.0017201256, 0.0021581065, 0.0027073042, 0.0033957870, 0.0042586071,
    0.0053394826, 0.0066928509, 0.0083863576, 0.0105038445, 0.0131488902,
    0.0164489307, 0.0205599431, 0.0256715863, 0.0320125562, 0.0398556989,
    0.0495221198, 0.0613831074, 0.0758581800, 0.0934070047, 0.1145124805,
    0.1396521834, 0.1692560327, 0.2036499335, 0.2429886272, 0.2871859014,
    0.3358556241, 0.3882805886, 0.4434251301, 0.5000000000, 0.5565748699,
    0.6117194114, 0.6641443759, 0.7128140986, 0.7570113728, 0.7963500665,
    0.8307439673, 0.8603478166, 0.8854875195, 0.9065929953, 0.9241418200,
    0.9386168926, 0.9504778802, 0.9601443011, 0.9679874438, 0.9743284137,
    0.9794400569, 0.9835510693, 0.9868511098, 0.9894961555, 0.9916136424,
    0.9933071491, 0.9946605174, 0.9957413929, 0.9966042130, 0.9972926958,
    0.9978418935, 0.9982798744, 0.9986290906, 0.9989074872, 0.9991293979,
    0.9993062655, 0.9994472214, 0.9995595498, 0.9996490604, 0.9997203853,
    0.9997772173, 0.9998225002, 0.9998585802, 0.9998873271, 0.9999102311,
    0.9999284795, 0.9999430185, 0.9999546021, 0.9999638310, 0.9999711838,
    0.9999770419, 0.9999817091, 0.9999854276, 0.9999883900, 0.9999907503,
    0.9999926307, 0.9999941289, 0.9999953224, 0.9999962734, 0.9999970310,
    0.9999976346, 0.9999981154, 0.9999984986, 0.9999988038, 0.9999990470,
    0.9999992407, 0.9999993951, 0.9999995181, 0.9999996160, 0.9999996941,
    0.9999997563, 0.9999998058, 0.9999998453, 0.9999998768, 0.9999999018,
    0.9999999218, 0.9999999377, 0.9999999503, 0.9999999604, 0.9999999685,
    0.9999999749, 0.9999999800, 0.9999999841, 0.9999999873, 0.9999999899,
    0.9999999919, 0.9999999936, 0.9999999949, 0.9999999959, 0.9999999968,
    0.9999999974, 0.9999999979};

template <typename T>
void ValidateLogisticGoldens(TfLiteTensor* tensors, const int tensor_count,
                             T* output_data, const T* golden,
                             int output_dims_count, float tolerance) {
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_LOGISTIC();
  micro::KernelRunner runner(registration, tensors, tensor_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

void TestLogisticFloat(int* input_dims_data, const float* input_data,
                       const float* golden, int* output_dims_data,
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

  ValidateLogisticGoldens(tensors, tensors_size, output_data, golden,
                          output_elements_count, 1e-5);
}

template <typename T>
void TestLogisticQuantized(int* input_dims_data, const float* input_data,
                           T* input_quantized, const float input_scale,
                           const int input_zero_point, const float* golden,
                           T* golden_quantized, int* output_dims_data,
                           const float output_scale,
                           const int output_zero_point, T* output_data,
                           float tolerance) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  tflite::Quantize(golden, golden_quantized, output_elements_count,
                   output_scale, output_zero_point);
  ValidateLogisticGoldens(tensors, tensors_size, output_data, golden_quantized,
                          output_elements_count, tolerance);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LogisticFloatBasicShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_basic];
  tflite::testing::TestLogisticFloat(
      tflite::testing::shape_basic, tflite::testing::input_data_basic,
      tflite::testing::golden_basic, tflite::testing::shape_basic, output_data);
}

TF_LITE_MICRO_TEST(LogisticQuantizedInt8BasicShouldMatchGolden) {
  const float input_scale = 0.1;
  const int input_zero_point = 0;
  int8_t input_quantized[tflite::testing::flat_size_basic];
  int8_t golden_quantized[tflite::testing::flat_size_basic];
  int8_t output_data[tflite::testing::flat_size_basic];

  tflite::testing::TestLogisticQuantized<int8_t>(
      tflite::testing::shape_basic, tflite::testing::input_data_basic,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::golden_basic, golden_quantized,
      tflite::testing::shape_basic,
      tflite::testing::quantized_output_scale_int8,
      tflite::testing::quantized_output_zero_point_int8, output_data, 1.0f);
}

TF_LITE_MICRO_TEST(LogisticFloatWideRangeShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_wide_range];
  tflite::testing::TestLogisticFloat(
      tflite::testing::shape_wide_range, tflite::testing::input_data_wide_range,
      tflite::testing::golden_wide_range, tflite::testing::shape_wide_range,
      output_data);
}

TF_LITE_MICRO_TEST(LogisticQuantizedInt8WideRangeShouldMatchGolden) {
  const float input_scale = 1.0;
  const int input_zero_point = 0;
  int8_t input_quantized[tflite::testing::flat_size_wide_range];
  int8_t golden_quantized[tflite::testing::flat_size_wide_range];
  int8_t output_data[tflite::testing::flat_size_wide_range];

  tflite::testing::TestLogisticQuantized<int8_t>(
      tflite::testing::shape_wide_range, tflite::testing::input_data_wide_range,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::golden_wide_range, golden_quantized,
      tflite::testing::shape_wide_range,
      tflite::testing::quantized_output_scale_int8,
      tflite::testing::quantized_output_zero_point_int8, output_data, 1.0f);
}

TF_LITE_MICRO_TEST(LogisticQuantizedInt16ShouldMatchGolden) {
  const float input_scale = 32.f / 65536.f;
  const int input_zero_point = 0;
  const float output_scale = 2.f / 65536.f;
  const int output_zero_point = 0;
  int16_t input_quantized[tflite::testing::int16_vec_size];
  int16_t golden_quantized[tflite::testing::int16_vec_size];
  int16_t output_data[tflite::testing::int16_vec_size];

  tflite::testing::TestLogisticQuantized<int16_t>(
      tflite::testing::shape_int16_vec, tflite::testing::int16_input_vec_fp,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::int16_golden_vec_fp, golden_quantized,
      tflite::testing::shape_int16_vec, output_scale, output_zero_point,
      output_data, 16.0f);
}

TF_LITE_MICRO_TESTS_END
