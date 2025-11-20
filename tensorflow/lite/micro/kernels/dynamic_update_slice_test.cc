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

constexpr int kNumInputs = 3;
constexpr int kNumOutputs = 1;
constexpr int kInputTensorIndex_0 = 0;
constexpr int kInputTensorIndex_1 = 1;
constexpr int kInputTensorIndex_2 = 2;
constexpr int kOutputTensorIndex = 3;

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

template <typename T, typename U>
void TestDynamicUpdateSlice(int* input_dims_data[kNumInputs],
                            const T* input_data_0, const T* input_data_1,
                            const U* input_data_2, const T* golden_data,
                            int* expected_dims, T* output_data) {
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
    TF_LITE_MICRO_EXPECT_EQ(golden_data[i], output_data[i]);
  }

  // check output dimensions (relocated) against original dimensions
  TF_LITE_MICRO_EXPECT_EQ(output_dims->size,
                          tensors[kOutputTensorIndex].dims->size);
  for (int i = 0; i < output_dims->size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(output_dims->data[i],
                            tensors[kOutputTensorIndex].dims->data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(DynamicUpdateSliceOpTestSimpleFloat) {
  int kInputDims_0[] = {2, 3, 3};
  int kInputDims_1[] = {2, 2, 1};
  int kInputDims_2[] = {1, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1,
                                                  kInputDims_2};
  int kOutputDims[] = {2, 3, 3};

  constexpr float kInput_0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  constexpr float kInput_1[] = {-1, -2};
  constexpr int32_t kInput_2[] = {1, 1};
  constexpr float kExpect[] = {1, 2, 3, 4, -1, 6, 7, -2, 9};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestDynamicUpdateSlice<float, int32_t>(
      kInputDims, kInput_0, kInput_1, kInput_2, kExpect, kOutputDims,
      output_data);
}

TF_LITE_MICRO_TEST(DynamicUpdateSliceOpTestSimpleInt8) {
  int kInputDims_0[] = {2, 3, 3};
  int kInputDims_1[] = {2, 2, 1};
  int kInputDims_2[] = {1, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1,
                                                  kInputDims_2};
  int kOutputDims[] = {2, 3, 3};

  constexpr int8_t kInput_0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  constexpr int8_t kInput_1[] = {-1, -2};
  constexpr int32_t kInput_2[] = {1, 1};
  constexpr int8_t kExpect[] = {1, 2, 3, 4, -1, 6, 7, -2, 9};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  int8_t output_data[kOutputCount];

  tflite::testing::TestDynamicUpdateSlice<int8_t, int32_t>(
      kInputDims, kInput_0, kInput_1, kInput_2, kExpect, kOutputDims,
      output_data);
}

TF_LITE_MICRO_TEST(DynamicUpdateSliceOpTestSimpleInt16) {
  int kInputDims_0[] = {2, 3, 3};
  int kInputDims_1[] = {2, 2, 1};
  int kInputDims_2[] = {1, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1,
                                                  kInputDims_2};
  int kOutputDims[] = {2, 3, 3};

  constexpr int16_t kInput_0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  constexpr int16_t kInput_1[] = {-1, -2};
  constexpr int32_t kInput_2[] = {1, 1};
  constexpr int16_t kExpect[] = {1, 2, 3, 4, -1, 6, 7, -2, 9};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  int16_t output_data[kOutputCount];

  tflite::testing::TestDynamicUpdateSlice<int16_t, int32_t>(
      kInputDims, kInput_0, kInput_1, kInput_2, kExpect, kOutputDims,
      output_data);
}

TF_LITE_MICRO_TEST(DynamicUpdateSliceOpTestSimpleInt32) {
  int kInputDims_0[] = {2, 3, 3};
  int kInputDims_1[] = {2, 2, 1};
  int kInputDims_2[] = {1, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1,
                                                  kInputDims_2};
  int kOutputDims[] = {2, 3, 3};

  constexpr int32_t kInput_0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  constexpr int32_t kInput_1[] = {-1, -2};
  constexpr int32_t kInput_2[] = {1, 1};
  constexpr int32_t kExpect[] = {1, 2, 3, 4, -1, 6, 7, -2, 9};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestDynamicUpdateSlice<int32_t, int32_t>(
      kInputDims, kInput_0, kInput_1, kInput_2, kExpect, kOutputDims,
      output_data);
}

TF_LITE_MICRO_TEST(DynamicUpdateSliceOpTestSimpleInt8IndicesI64) {
  int kInputDims_0[] = {2, 3, 3};
  int kInputDims_1[] = {2, 2, 1};
  int kInputDims_2[] = {1, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1,
                                                  kInputDims_2};
  int kOutputDims[] = {2, 3, 3};

  constexpr int8_t kInput_0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  constexpr int8_t kInput_1[] = {-1, -2};
  constexpr int64_t kInput_2[] = {1, 1};
  constexpr int8_t kExpect[] = {1, 2, 3, 4, -1, 6, 7, -2, 9};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  int8_t output_data[kOutputCount];

  tflite::testing::TestDynamicUpdateSlice<int8_t, int64_t>(
      kInputDims, kInput_0, kInput_1, kInput_2, kExpect, kOutputDims,
      output_data);
}

TF_LITE_MICRO_TEST(DynamicUpdateSliceOpTestBoundaryTest) {
  int kInputDims_0[] = {2, 3, 3};
  int kInputDims_1[] = {2, 2, 2};
  int kInputDims_2[] = {1, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1,
                                                  kInputDims_2};
  int kOutputDims[] = {2, 3, 3};

  constexpr float kInput_0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  constexpr float kInput_1[] = {-1, -2, -3, -4};
  constexpr int32_t kInput_2[] = {2, 2};
  constexpr float kExpect[] = {1, 2, 3, 4, -1, -2, 7, -3, -4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestDynamicUpdateSlice<float, int32_t>(
      kInputDims, kInput_0, kInput_1, kInput_2, kExpect, kOutputDims,
      output_data);
}
TF_LITE_MICRO_TESTS_END
