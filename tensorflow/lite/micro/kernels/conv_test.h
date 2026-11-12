/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_CONV_TEST_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_CONV_TEST_H_

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {

using micro::kConvBiasTensor;
using micro::kConvInputTensor;
using micro::kConvOutputTensor;
using micro::kConvQuantizedDimension;
using micro::kConvWeightsTensor;

constexpr int kConvMaxTensors = 4;
constexpr int kConvMaxInputTensors = 3;

template <typename T>
TfLiteStatus InvokeConv(TfLiteTensor* tensors, int tensors_size,
                        int output_length, const TfLiteConvParams* conv_params,
                        TFLMRegistration registration, T* output_data) {
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, conv_params);
  const char* init_data = reinterpret_cast<const char*>(conv_params);
  TfLiteStatus status = runner.InitAndPrepare(init_data);
  if (status != kTfLiteOk) {
    return status;
  }
  return runner.Invoke();
}

template <typename T>
void ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                         const T* expected_output_data, int output_length,
                         const TfLiteConvParams* conv_params,
                         TFLMRegistration registration, T* output_data,
                         float tolerance = 1e-5) {
  TfLiteStatus status = InvokeConv(tensors, tensors_size, output_length,
                                   conv_params, registration, output_data);
  ASSERT_EQ(status, kTfLiteOk);
  for (int i = 0; i < output_length; ++i) {
    EXPECT_NEAR(expected_output_data[i], output_data[i], tolerance);
  }
}

void ValidateConvFailsDuringPrepare(TfLiteTensor* tensors, int tensors_size,
                                    const TfLiteConvParams* conv_params,
                                    TFLMRegistration registration,
                                    void* output_data);

void TestConvFloat(int* input_dims_data, const float* input_data,
                   int* filter_dims_data, const float* filter_data,
                   int* bias_dims_data, const float* bias_data,
                   int* output_dims_data, const float* expected_output_data,
                   TfLiteConvParams* conv_params, TFLMRegistration registration,
                   float* output_data);

void TestConvQuantizedPerChannel(
    int* input_dims_data, const float* input_data, int8_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    int* bias_dims_data, const float* bias_data, int32_t* bias_data_quantized,
    float* bias_scales, int* bias_zero_points, int* output_dims_data,
    const float* expected_output_data, int8_t* expected_output_data_quantized,
    float output_scale, int output_zero_point, TfLiteConvParams* conv_params,
    TFLMRegistration registration, int8_t* output_data,
    TfLiteType tensor_weight_type = kTfLiteNoType);

void TestConvQuantizedPerChannel(
    int* input_dims_data, const float* input_data, int16_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    int* bias_dims_data, const float* bias_data,
    std::int64_t* bias_data_quantized, float* bias_scales,
    int* bias_zero_points, int* output_dims_data,
    const float* expected_output_data, int16_t* expected_output_data_quantized,
    float output_scale, int output_zero_point, TfLiteConvParams* conv_params,
    TFLMRegistration registration, int16_t* output_data);

void TestConvQuantizedPerChannel(
    int* input_dims_data, const float* input_data, int16_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    int* bias_dims_data, const float* bias_data, int32_t* bias_data_quantized,
    float* bias_scales, int* bias_zero_points, int* output_dims_data,
    const float* expected_output_data, int16_t* expected_output_data_quantized,
    float output_scale, int output_zero_point, TfLiteConvParams* conv_params,
    TFLMRegistration registration, int16_t* output_data);

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_CONV_TEST_H_
