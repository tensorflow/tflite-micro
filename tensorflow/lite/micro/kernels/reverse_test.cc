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
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kNumInputs = 2;
constexpr int kNumOutputs = 1;
constexpr int kInputTensorIndex_0 = 0;
constexpr int kInputTensorIndex_1 = 1;
constexpr int kOutputTensorIndex = 2;

void ExecuteReverseTest(TfLiteTensor* tensors, int tensors_count) {
  int kInputArrayData[] = {kNumInputs, kInputTensorIndex_0,
                           kInputTensorIndex_1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  int kOutputArrayData[] = {kNumOutputs, kOutputTensorIndex};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TFLMRegistration registration = tflite::Register_REVERSE_V2();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestReverse(int* input_dims_data[kNumInputs], const T* input_data_0,
                 const int32_t* input_data_1, int* expected_dims,
                 const T* expected_data, T* output_data) {
  TfLiteIntArray* input_dims_0 = IntArrayFromInts(input_dims_data[0]);
  TfLiteIntArray* input_dims_1 = IntArrayFromInts(input_dims_data[1]);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  TfLiteTensor tensors[] = {
      CreateTensor(input_data_0, input_dims_0),
      CreateTensor(input_data_1, input_dims_1),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  ExecuteReverseTest(tensors, tensors_count);

  // check output data against expected
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], 0);
  }

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

// float32 tests.
TF_LITE_MICRO_TEST(ReverseOpTestFloatOneDimension) {
  int kInputDims_0[] = {1, 4};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {1, 4};

  const float kInput_0[] = {1, 2, 3, 4};
  const int32_t kInput_1[] = {0};
  const float kExpect[] = {4, 3, 2, 1};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestFloatMultiDimensions) {
  int kInputDims_0[] = {3, 4, 3, 2};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 4, 3, 2};

  const float kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  const int32_t kInput_1[] = {1};
  const float kExpect[] = {5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                           17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

// int32 tests
TF_LITE_MICRO_TEST(ReverseOpTestInt32OneDimension) {
  int kInputDims_0[] = {1, 4};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {1, 4};

  const int32_t kInput_0[] = {1, 2, 3, 4};
  const int32_t kInput_1[] = {0};
  const int32_t kExpect[] = {4, 3, 2, 1};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt32MultiDimensions) {
  int kInputDims_0[] = {3, 4, 3, 2};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 4, 3, 2};

  const int32_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  const int32_t kInput_1[] = {1};
  const int32_t kExpect[] = {5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                             17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt32MultiDimensionsFirst) {
  int kInputDims_0[] = {3, 3, 3, 3};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 3, 3};

  const int32_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 16, 17, 18,
                              19, 20, 21, 22, 23, 24, 25, 26, 27};
  const int32_t kInput_1[] = {0};
  const int32_t kExpect[] = {19, 20, 21, 22, 23, 24, 25, 26, 27,
                             10, 11, 12, 13, 14, 15, 16, 17, 18,
                             1,  2,  3,  4,  5,  6,  7,  8,  9};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt32MultiDimensionsSecond) {
  int kInputDims_0[] = {3, 3, 3, 3};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 3, 3};

  const int32_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 16, 17, 18,
                              19, 20, 21, 22, 23, 24, 25, 26, 27};
  const int32_t kInput_1[] = {1};
  const int32_t kExpect[] = {7,  8,  9,  4,  5,  6,  1,  2,  3,
                             16, 17, 18, 13, 14, 15, 10, 11, 12,
                             25, 26, 27, 22, 23, 24, 19, 20, 21};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt32MultiDimensionsThird) {
  int kInputDims_0[] = {3, 3, 3, 3};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 3, 3};

  const int32_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 16, 17, 18,
                              19, 20, 21, 22, 23, 24, 25, 26, 27};
  const int32_t kInput_1[] = {2};
  const int32_t kExpect[] = {3,  2,  1,  6,  5,  4,  9,  8,  7,
                             12, 11, 10, 15, 14, 13, 18, 17, 16,
                             21, 20, 19, 24, 23, 22, 27, 26, 25};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt32MultiDimensionsFirstSecond) {
  int kInputDims_0[] = {3, 3, 3, 3};
  int kInputDims_1[] = {1, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 3, 3};

  const int32_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 16, 17, 18,
                              19, 20, 21, 22, 23, 24, 25, 26, 27};
  const int32_t kInput_1[] = {0, 1};

  const int32_t kExpect[] = {25, 26, 27, 22, 23, 24, 19, 20, 21,
                             16, 17, 18, 13, 14, 15, 10, 11, 12,
                             7,  8,  9,  4,  5,  6,  1,  2,  3};

  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt32MultiDimensionsSecondThird) {
  int kInputDims_0[] = {3, 3, 3, 3};
  int kInputDims_1[] = {1, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 3, 3};

  const int32_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 16, 17, 18,
                              19, 20, 21, 22, 23, 24, 25, 26, 27};
  const int32_t kInput_1[] = {1, 2};

  const int32_t kExpect[] = {9,  8,  7,  6,  5,  4,  3,  2,  1,
                             18, 17, 16, 15, 14, 13, 12, 11, 10,
                             27, 26, 25, 24, 23, 22, 21, 20, 19};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt32MultiDimensionsSecondFirst) {
  int kInputDims_0[] = {3, 3, 3, 3};
  int kInputDims_1[] = {1, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 3, 3};

  const int32_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 16, 17, 18,
                              19, 20, 21, 22, 23, 24, 25, 26, 27};
  const int32_t kInput_1[] = {1, 0};

  const int32_t kExpect[] = {25, 26, 27, 22, 23, 24, 19, 20, 21,
                             16, 17, 18, 13, 14, 15, 10, 11, 12,
                             7,  8,  9,  4,  5,  6,  1,  2,  3};

  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt32MultiDimensionsAll) {
  int kInputDims_0[] = {3, 3, 3, 3};
  int kInputDims_1[] = {1, 3};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 3, 3};

  const int32_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 16, 17, 18,
                              19, 20, 21, 22, 23, 24, 25, 26, 27};
  const int32_t kInput_1[] = {0, 1, 2};
  const int32_t kExpect[] = {27, 26, 25, 24, 23, 22, 21, 20, 19,
                             18, 17, 16, 15, 14, 13, 12, 11, 10,
                             9,  8,  7,  6,  5,  4,  3,  2,  1};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

// uint8 tests
TF_LITE_MICRO_TEST(ReverseOpTestUint8OneDimension) {
  int kInputDims_0[] = {1, 4};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {1, 4};

  const uint8_t kInput_0[] = {1, 2, 3, 4};
  const int32_t kInput_1[] = {0};
  const uint8_t kExpect[] = {4, 3, 2, 1};

  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  uint8_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestUint8MultiDimensions) {
  int kInputDims_0[] = {3, 4, 3, 2};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 4, 3, 2};

  const uint8_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  const int32_t kInput_1[] = {1};
  const uint8_t kExpect[] = {5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                             17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  uint8_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

// int8 tests
TF_LITE_MICRO_TEST(ReverseOpTestInt8OneDimension) {
  int kInputDims_0[] = {1, 4};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {1, 4};

  const int8_t kInput_0[] = {1, 2, -1, -2};
  const int32_t kInput_1[] = {0};
  const int8_t kExpect[] = {-2, -1, 2, 1};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int8_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt8MultiDimensions) {
  int kInputDims_0[] = {3, 4, 3, 2};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 4, 3, 2};

  const int8_t kInput_0[] = {-1, -2, -3, -4, 5,   6,   7,   8,
                             9,  10, 11, 12, 13,  14,  15,  16,
                             17, 18, 19, 20, -21, -22, -23, -24};
  const int32_t kInput_1[] = {1};
  const int8_t kExpect[] = {5,  6,  -3, -4, -1, -2, 11,  12,  9,   10,  7,  8,
                            17, 18, 15, 16, 13, 14, -23, -24, -21, -22, 19, 20};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int8_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

// int16 tests
TF_LITE_MICRO_TEST(ReverseOpTestInt16OneDimension) {
  int kInputDims_0[] = {1, 4};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {1, 4};

  const int16_t kInput_0[] = {1, 2, 3, 4};
  const int32_t kInput_1[] = {0};
  const int16_t kExpect[] = {4, 3, 2, 1};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int16_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt16MultiDimensions) {
  int kInputDims_0[] = {3, 4, 3, 2};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 4, 3, 2};

  const int16_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  const int32_t kInput_1[] = {1};
  const int16_t kExpect[] = {5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                             17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int16_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

// int64 tests
TF_LITE_MICRO_TEST(ReverseOpTestInt64OneDimension) {
  int kInputDims_0[] = {1, 4};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {1, 4};

  const int64_t kInput_0[] = {1, 2, 3, 4};
  const int32_t kInput_1[] = {0};
  const int64_t kExpect[] = {4, 3, 2, 1};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int64_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TEST(ReverseOpTestInt64MultiDimensions) {
  int kInputDims_0[] = {3, 4, 3, 2};
  int kInputDims_1[] = {1, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 4, 3, 2};

  const int64_t kInput_0[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  const int32_t kInput_1[] = {1};
  const int64_t kExpect[] = {5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                             17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20};
  const int kOutputCount = std::extent<decltype(kExpect)>::value;
  int64_t output_data[kOutputCount];

  tflite::testing::TestReverse(kInputDims, kInput_0, kInput_1, kOutputDims,
                               kExpect, output_data);
}

TF_LITE_MICRO_TESTS_END
