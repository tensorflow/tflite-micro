/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <iterator>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr float kTestTolerance = 7.41e-03;
constexpr int kNumInputs = 2;
constexpr int kNumOutputs = 1;
constexpr int kInputTensorIndex_0 = 0;
constexpr int kInputTensorIndex_1 = 1;
constexpr int kOutputTensorIndex = 2;

// min/max are used to compute scale, zero-point is 0
template <size_t kInputSize>
struct TestEmbeddingLookupParams {
  // quantization parameters
  float data_min;                 // input data minimum value
  float data_max;                 // input data maximum value
  int8_t input_data[kInputSize];  // quantized input storage
};

void ExecuteEmbeddingLookupTest(TfLiteTensor* tensors, int tensors_count) {
  int kInputArrayData[] = {kNumInputs, kInputTensorIndex_0,
                           kInputTensorIndex_1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  int kOutputArrayData[] = {kNumOutputs, kOutputTensorIndex};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TFLMRegistration registration = tflite::Register_EMBEDDING_LOOKUP();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <size_t N>
void TestEmbeddingLookupQuantized(TestEmbeddingLookupParams<N>& params,
                                  int* input_dims_data[kNumInputs],
                                  const int32_t* input_data_0,
                                  const float* input_data_1, int* expected_dims,
                                  const float* expected_data,
                                  float* output_data) {
  TfLiteIntArray* input_dims_0 = IntArrayFromInts(input_dims_data[0]);
  TfLiteIntArray* input_dims_1 = IntArrayFromInts(input_dims_data[1]);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  const float scale =
      SymmetricScaleFromMinMax<int8_t>(params.data_min, params.data_max);

  TfLiteTensor tensors[] = {
      CreateTensor(input_data_0, input_dims_0),
      CreateQuantizedTensor(input_data_1, params.input_data, input_dims_1,
                            scale, 0),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  ExecuteEmbeddingLookupTest(tensors, tensors_count);

  // check output data against expected
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTestTolerance);
  }

  // check output dimensions (relocated) against original dimensions
  TF_LITE_MICRO_EXPECT_EQ(output_dims->size,
                          tensors[kOutputTensorIndex].dims->size);
  for (int i = 0; i < output_dims->size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(output_dims->data[i],
                            tensors[kOutputTensorIndex].dims->data[i]);
  }
}  // namespace

template <typename T>
void TestEmbeddingLookup(int* input_dims_data[kNumInputs],
                         const int32_t* input_data_0, const T* input_data_1,
                         int* expected_dims, const T* expected_data,
                         T* output_data) {
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
  ExecuteEmbeddingLookupTest(tensors, tensors_count);

  // check output data against expected
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTestTolerance);
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

TF_LITE_MICRO_TEST(EmbeddingLookupOpTestSimpleFloat) {
  int kInputDims_0[] = {1, 3};
  int kInputDims_1[] = {3, 3, 2, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 2, 4};

  constexpr int32_t kInput_0[] = {1, 0, 2};
  constexpr float kInput_1[] = {
      0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr float kExpect[] = {
      1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestEmbeddingLookup(kInputDims, kInput_0, kInput_1,
                                       kOutputDims, kExpect, output_data);
}

TF_LITE_MICRO_TEST(HybridEmbeddingLookupHybridOpTestSimple2DTestInt8) {
  int kInputDims_0[] = {1, 3};
  int kInputDims_1[] = {2, 3, 8};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {2, 3, 8};

  constexpr int32_t kInput_0[] = {1, 0, 2};
  constexpr float kInput_1[] = {
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kInputCount_1 = std::extent<decltype(kInput_1)>::value;
  constexpr float kExpect[] = {
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestEmbeddingLookupParams<kInputCount_1> params = {};
  auto minmax = std::minmax_element(std::begin(kInput_1), std::end(kInput_1));
  params.data_max = *minmax.second;
  params.data_min = *minmax.first;

  tflite::testing::TestEmbeddingLookupQuantized(params, kInputDims, kInput_0,
                                                kInput_1, kOutputDims, kExpect,
                                                output_data);
}

TF_LITE_MICRO_TEST(HybridEmbeddingLookupHybridOpTestSimple3DTestInt8) {
  int kInputDims_0[] = {1, 3};
  int kInputDims_1[] = {3, 3, 2, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 2, 4};

  constexpr int32_t kInput_0[] = {1, 0, 2};
  constexpr float kInput_1[] = {
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kInputCount_1 = std::extent<decltype(kInput_1)>::value;
  constexpr float kExpect[] = {
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestEmbeddingLookupParams<kInputCount_1> params = {};
  auto minmax = std::minmax_element(std::begin(kInput_1), std::end(kInput_1));
  params.data_max = *minmax.second;
  params.data_min = *minmax.first;

  tflite::testing::TestEmbeddingLookupQuantized(params, kInputDims, kInput_0,
                                                kInput_1, kOutputDims, kExpect,
                                                output_data);
}

TF_LITE_MICRO_TEST(HybridEmbeddingLookupHybridOpTestSimple4DTestInt8) {
  int kInputDims_0[] = {1, 3};
  int kInputDims_1[] = {4, 3, 2, 2, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {4, 3, 2, 2, 2};

  constexpr int32_t kInput_0[] = {1, 0, 2};
  constexpr float kInput_1[] = {
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kInputCount_1 = std::extent<decltype(kInput_1)>::value;
  constexpr float kExpect[] = {
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestEmbeddingLookupParams<kInputCount_1> params = {};
  auto minmax = std::minmax_element(std::begin(kInput_1), std::end(kInput_1));
  params.data_max = *minmax.second;
  params.data_min = *minmax.first;

  tflite::testing::TestEmbeddingLookupQuantized(params, kInputDims, kInput_0,
                                                kInput_1, kOutputDims, kExpect,
                                                output_data);
}

TF_LITE_MICRO_TEST(EmbeddingLookupOpTestSimpleInt8) {
  int kInputDims_0[] = {1, 3};
  int kInputDims_1[] = {3, 3, 2, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 2, 4};

  constexpr int32_t kInput_0[] = {1, 0, 2};
  constexpr int8_t kInput_1[] = {
      0,   1,   2,   3,   10,  11,  12,  13,   // Row 0
      100, 101, 102, 103, 110, 111, 112, 113,  // Row 1
      -56, -55, -54, -53, -46, -45, -44, -43,  // Row 2
  };
  constexpr int8_t kExpect[] = {
      100, 101, 102, 103, 110, 111, 112, 113,  // Row 1
      0,   1,   2,   3,   10,  11,  12,  13,   // Row 0
      -56, -55, -54, -53, -46, -45, -44, -43,  // Row 2
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  int8_t output_data[kOutputCount];

  tflite::testing::TestEmbeddingLookup(kInputDims, kInput_0, kInput_1,
                                       kOutputDims, kExpect, output_data);
}

TF_LITE_MICRO_TESTS_END
