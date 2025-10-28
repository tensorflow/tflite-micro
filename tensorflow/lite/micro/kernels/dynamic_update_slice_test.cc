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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// constexpr float kTestTolerance = 7.41e-03;
constexpr int kNumInputs = 3;
constexpr int kNumOutputs = 1;
constexpr int kInputTensorIndex_0 = 0;
constexpr int kInputTensorIndex_1 = 1;
constexpr int kInputTensorIndex_2 = 2;
constexpr int kOutputTensorIndex = 3;

// min/max are used to compute scale, zero-point is 0
template <size_t kInputSize>
struct TestDynamicUpdateSliceParams {
  // quantization parameters
  float data_min;                 // input data minimum value
  float data_max;                 // input data maximum value
  int8_t input1_data[kInputSize];  // quantized input storage
  int8_t input2_data[kInputSize];  // quantized input storage
};

void ExecuteDynamicUpdateSliceTest(TfLiteTensor* tensors, int tensors_count) {
  int kInputArrayData[] = {kNumInputs, kInputTensorIndex_0, kInputTensorIndex_1,
                           kInputTensorIndex_2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  int kOutputArrayData[] = {kNumOutputs, kOutputTensorIndex};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TFLMRegistration registration = tflite::Register_DYNAMIC_UPDATE_SLICE();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

void TestDynamicUpdateSliceFloat(int* input_dims_data[kNumInputs],
                                 const float* input_data_0,
                                 const float* input_data_1,
                                 const int32_t* input_data_2,
                                 const float* golden_data, int* expected_dims,
                                 float* output_data) {
  TfLiteIntArray* input_dims_0 = IntArrayFromInts(input_dims_data[0]);
  TfLiteIntArray* input_dims_1 = IntArrayFromInts(input_dims_data[1]);
  TfLiteIntArray* input_dims_2 = IntArrayFromInts(input_dims_data[2]);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  TfLiteTensor tensors[] = {
      CreateTensor(input_data_0, input_dims_0),
      CreateTensor(input_data_1, input_dims_1),
      CreateTensor(input_data_2, input_dims_2),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  ExecuteDynamicUpdateSliceTest(tensors, tensors_count);

  // check output data against expected
  for (int i = 0; i < output_count; i++) {
    printf("output_data[%d] = %f\n", i, output_data[i]);
    TF_LITE_MICRO_EXPECT_NEAR(golden_data[i], output_data[i], 0.0);
  }

  // check output dimensions (relocated) against original dimensions
  TF_LITE_MICRO_EXPECT_EQ(output_dims->size,
                          tensors[kOutputTensorIndex].dims->size);
  for (int i = 0; i < output_dims->size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(output_dims->data[i],
                            tensors[kOutputTensorIndex].dims->data[i]);
  }
}

// TODO(rameshkunasi): Add quantized test for dynamic update slice.

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(DynamicUpdateSliceOpTestSimpleFloat) {
  int32_t kInputDims_0[] = {2, 3, 3};
  int32_t kInputDims_1[] = {2, 2, 1};
  int32_t kInputDims_2[] = {1, 2};
  int32_t* kInputDims[tflite::testing::kNumInputs] = {
      kInputDims_0, kInputDims_1, kInputDims_2};
  int kOutputDims[] = {2, 3, 3};

  constexpr float kInput_0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  constexpr float kInput_1[] = {-1, -2};
  constexpr int32_t kInput_2[] = {1, 1};
  constexpr float kExpect[] = {1, 2, 3, 4, -1, 6, 7, -2, 9};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestDynamicUpdateSliceFloat(kInputDims, kInput_0, kInput_1,
                                               kInput_2, kExpect, kOutputDims,
                                               output_data);
}

TF_LITE_MICRO_TEST(DynamicUpdateSliceOpTestBoundaryTest) {
  int32_t kInputDims_0[] = {2, 3, 3};
  int32_t kInputDims_1[] = {2, 2, 2};
  int32_t kInputDims_2[] = {1, 2};
  int32_t* kInputDims[tflite::testing::kNumInputs] = {
      kInputDims_0, kInputDims_1, kInputDims_2};
  int kOutputDims[] = {2, 3, 3};

  constexpr float kInput_0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  constexpr float kInput_1[] = {-1, -2, -3, -4};
  constexpr int32_t kInput_2[] = {2, 2};
  constexpr float kExpect[] = {1, 2, 3, 4, -1, -2, 7, -3, -4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestDynamicUpdateSliceFloat(kInputDims, kInput_0, kInput_1,
                                               kInput_2, kExpect, kOutputDims,
                                               output_data);
}
TF_LITE_MICRO_TESTS_END

