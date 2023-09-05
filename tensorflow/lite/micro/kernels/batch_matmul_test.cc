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
#include <numeric>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr float kFloatTolerance = 1e-5;

constexpr int kNumInputs = 2;
constexpr int kNumOutputs = 1;
constexpr int kLhsInputTensorIndex = 0;
constexpr int kRhsInputTensorIndex = 1;
constexpr int kOutputTensorIndex = 2;

// data_min/data_max are used to compute symmetric scale, zero-point is 0
// scale should be 0 to use data_min/data_max
template <typename T, size_t kNumElements>
struct TestQuantizationParams {
  // quantization parameters
  float scale;  // if 0, use data_min and data_max
  int zero_point;
  float data_min;  // input data minimum value
  float data_max;  // input data maximum value

  T quantized_data[kNumElements];  // quantized storage
};

micro::KernelRunner* GetKernelRunnerInstance(
    TfLiteTensor* tensors, int tensors_count,
    const TfLiteBatchMatMulParams& params, bool need_init_prepare) {
  static int kInputArrayData[] = {kNumInputs, kLhsInputTensorIndex,
                                  kRhsInputTensorIndex};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  static int kOutputArrayData[] = {kNumOutputs, kOutputTensorIndex};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  static const TFLMRegistration registration = tflite::Register_BATCH_MATMUL();

  alignas(micro::KernelRunner) static char
      kernel_runner_buffer[sizeof(micro::KernelRunner)] = {};

  static micro::KernelRunner* runner = nullptr;
  if (runner == nullptr || need_init_prepare) {
    runner = new (kernel_runner_buffer)
        micro::KernelRunner(registration, tensors, tensors_count, inputs_array,
                            outputs_array, &params);

    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner->InitAndPrepare());
  }

  return runner;
}

void TestBatchMatMulFloat(const TfLiteBatchMatMulParams& params,
                          const int* input_dims_data[kNumInputs],
                          const float* input_data_lhs,
                          const float* input_data_rhs, const int* expected_dims,
                          const float* expected_data, float* output_data,
                          bool need_constant_rhs = false,
                          bool need_init_prepare = true) {
  TfLiteIntArray* input_dims_lhs = IntArrayFromInts(input_dims_data[0]);
  TfLiteIntArray* input_dims_rhs = IntArrayFromInts(input_dims_data[1]);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int kOutputCount = ElementCount(*output_dims);

  static TfLiteTensor tensors[kNumInputs + kNumOutputs];

  if (need_init_prepare) {
    tensors[kLhsInputTensorIndex] =
        CreateTensor(input_data_lhs, input_dims_lhs);
    tensors[kRhsInputTensorIndex] =
        CreateTensor(input_data_rhs, input_dims_rhs);
    if (need_constant_rhs) {
      tensors[kRhsInputTensorIndex].allocation_type = kTfLiteMmapRo;
    }
    tensors[kOutputTensorIndex] = CreateTensor(output_data, output_dims);
  }

  constexpr int kTensorCount = std::extent<decltype(tensors)>::value;
  micro::KernelRunner* runner =
      GetKernelRunnerInstance(tensors, kTensorCount, params, need_init_prepare);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner->Invoke());

  // check output data against expected
  for (int i = 0; i < kOutputCount; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i],
                              kFloatTolerance);
  }

  // check output dimensions (relocated) against original dimensions
  TF_LITE_MICRO_EXPECT_EQ(output_dims->size,
                          tensors[kOutputTensorIndex].dims->size);
  for (int i = 0; i < output_dims->size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(output_dims->data[i],
                            tensors[kOutputTensorIndex].dims->data[i]);
  }
}

template <typename T, size_t kNumElements>
void SetScaleAndZeroPoint(TestQuantizationParams<T, kNumElements>* q_params) {
  if (q_params->scale == 0.0f || q_params->data_max != 0 ||
      q_params->data_min != 0) {
    q_params->scale =
        ScaleFromMinMax<T>(q_params->data_min, q_params->data_max);
    q_params->zero_point =
        ZeroPointFromMinMax<T>(q_params->data_min, q_params->data_max);
  }
}

template <typename T, size_t kNumLhs, size_t kNumRhs, size_t kNumOutput>
void TestBatchMatMulQuantized(
    const TfLiteBatchMatMulParams& params,
    TestQuantizationParams<T, kNumLhs>* quantization_lhs,
    TestQuantizationParams<T, kNumRhs>* quantization_rhs,
    TestQuantizationParams<T, kNumOutput>* quantization_output,
    const int* input_dims_data[kNumInputs], const float* input_data_lhs,
    const float* input_data_rhs, const int* expected_dims,
    const T* expected_data, const float* output_data) {
  TfLiteIntArray* input_dims_lhs = IntArrayFromInts(input_dims_data[0]);
  TfLiteIntArray* input_dims_rhs = IntArrayFromInts(input_dims_data[1]);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int kOutputCount = ElementCount(*output_dims);

  static TfLiteTensor tensors[kNumInputs + kNumOutputs];

  SetScaleAndZeroPoint<T, kNumLhs>(quantization_lhs);
  tensors[kLhsInputTensorIndex] = CreateQuantizedTensor(
      input_data_lhs, quantization_lhs->quantized_data, input_dims_lhs,
      quantization_lhs->scale, quantization_lhs->zero_point);
  SetScaleAndZeroPoint<T, kNumRhs>(quantization_rhs);
  tensors[kRhsInputTensorIndex] = CreateQuantizedTensor(
      input_data_rhs, quantization_rhs->quantized_data, input_dims_rhs,
      quantization_rhs->scale, quantization_rhs->zero_point);
  SetScaleAndZeroPoint<T, kNumOutput>(quantization_output);
  tensors[kOutputTensorIndex] = CreateQuantizedTensor(
      quantization_output->quantized_data, output_dims,
      quantization_output->scale, quantization_output->zero_point);

  constexpr int kTensorCount = std::extent<decltype(tensors)>::value;
  micro::KernelRunner* runner =
      GetKernelRunnerInstance(tensors, kTensorCount, params, true);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner->Invoke());

  // check output data against expected
  for (int i = 0; i < kOutputCount; i++) {
    TF_LITE_MICRO_EXPECT_EQ(expected_data[i],
                            quantization_output->quantized_data[i]);
  }
  // check dequantized output data against expected
  for (int i = 0; i < kOutputCount; i++) {
    float dequantized_value = (quantization_output->quantized_data[i] -
                               quantization_output->zero_point) *
                              quantization_output->scale;
    TF_LITE_MICRO_EXPECT_NEAR(output_data[i], dequantized_value,
                              kFloatTolerance);
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

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Ones) {
  constexpr int kLhsInputDims[] = {4, 3, 2, 1, 4};
  constexpr int kRhsInputDims[] = {4, 3, 1, 4, 1};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr size_t kLhsInputSize = 24;
  float lhs_input[kLhsInputSize];
  std::iota(std::begin(lhs_input), std::end(lhs_input), 1);

  constexpr size_t kRhsInputSize = 12;
  float rhs_input[kRhsInputSize];
  std::iota(std::begin(rhs_input), std::end(rhs_input), 1);

  constexpr float kExpect[] = {30, 70, 278, 382, 782, 950};
  constexpr int kOutputDims[] = {4, 3, 2, 1, 1};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, lhs_input,
                                        rhs_input, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Flatten) {
  constexpr int kLhsInputDims[] = {4, 3, 2, 2, 4};
  constexpr int kRhsInputDims[] = {4, 3, 1, 4, 1};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr size_t kLhsInputSize = 48;
  float lhs_input[kLhsInputSize];
  std::iota(std::begin(lhs_input), std::end(lhs_input), 1);

  constexpr size_t kRhsInputSize = 12;
  float rhs_input[kRhsInputSize];
  std::iota(std::begin(rhs_input), std::end(rhs_input), 1);

  constexpr float kExpect[] = {30,  70,  110,  150,  486,  590,
                               694, 798, 1454, 1622, 1790, 1958};
  constexpr int kOutputDims[] = {4, 3, 2, 2, 1};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, lhs_input,
                                        rhs_input, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Simple) {
  constexpr int kLhsInputDims[] = {3, 1, 2, 3};
  constexpr int kRhsInputDims[] = {3, 1, 3, 4};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr size_t kLhsInputSize = 6;
  float lhs_input[kLhsInputSize];
  std::iota(std::begin(lhs_input), std::end(lhs_input), 1);

  constexpr size_t kRhsInputSize = 12;
  float rhs_input[kRhsInputSize];
  std::iota(std::begin(rhs_input), std::end(rhs_input), 7);

  constexpr float kExpect[] = {74., 80., 86., 92., 173., 188., 203., 218.};
  constexpr int kOutputDims[] = {3, 1, 2, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, lhs_input,
                                        rhs_input, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_SimpleRHSAdjoint) {
  constexpr int kLhsInputDims[] = {3, 1, 2, 3};
  constexpr int kRhsInputDims[] = {3, 1, 4, 3};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr size_t kLhsInputSize = 6;
  float lhs_input[kLhsInputSize];
  std::iota(std::begin(lhs_input), std::end(lhs_input), 1);

  constexpr float kRhsInput[] = {7, 11, 15, 8, 12, 16, 9, 13, 17, 10, 14, 18};

  constexpr float kExpect[] = {74., 80., 86., 92., 173., 188., 203., 218.};
  constexpr int kOutputDims[] = {3, 1, 2, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      true,   // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, lhs_input,
                                        kRhsInput, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_SimpleLHSAdjoint) {
  constexpr int kLhsInputDims[] = {3, 1, 3, 2};
  constexpr int kRhsInputDims[] = {3, 1, 3, 4};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};
  constexpr float kLhsInput[] = {1, 4, 2, 5, 3, 6};

  constexpr size_t kRhsInputSize = 12;
  float rhs_input[kRhsInputSize];
  std::iota(std::begin(rhs_input), std::end(rhs_input), 7);

  constexpr float kExpect[] = {74., 80., 86., 92., 173., 188., 203., 218.};
  constexpr int kOutputDims[] = {3, 1, 2, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      true,   // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kLhsInput,
                                        rhs_input, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_BatchSizeTwo) {
  constexpr int kLhsInputDims[] = {3, 2, 2, 3};
  constexpr int kRhsInputDims[] = {3, 2, 3, 4};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};
  constexpr size_t kLhsInputSize = 12;
  float lhs_input[kLhsInputSize];
  std::iota(std::begin(lhs_input), std::end(lhs_input), 1);

  constexpr size_t kRhsInputSize = 24;
  float rhs_input[kRhsInputSize];
  std::iota(std::begin(rhs_input), std::end(rhs_input), 7);

  constexpr float kExpect[] = {74.,  80.,  86.,  92.,  173., 188., 203., 218.,
                               560., 584., 608., 632., 767., 800., 833., 866.};
  constexpr int kOutputDims[] = {3, 2, 2, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, lhs_input,
                                        rhs_input, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast) {
  constexpr int kLhsInputDims[] = {3, 2, 2, 3};
  constexpr int kRhsInputDims[] = {2, 3, 4};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};
  constexpr size_t kLhsInputSize = 12;
  float lhs_input[kLhsInputSize];
  std::iota(std::begin(lhs_input), std::end(lhs_input), 1);

  constexpr size_t kRhsInputSize = 12;
  float rhs_input[kRhsInputSize];
  std::iota(std::begin(rhs_input), std::end(rhs_input), 7);

  constexpr float kExpect[] = {74.,  80.,  86.,  92.,  173., 188., 203., 218.,
                               272., 296., 320., 344., 371., 404., 437., 470.};
  constexpr int kOutputDims[] = {3, 2, 2, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, lhs_input,
                                        rhs_input, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_BroadcastLHSAdjoint) {
  constexpr int kLhsInputDims[] = {3, 2, 3, 2};
  constexpr int kRhsInputDims[] = {2, 3, 4};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr float kLhsInput[] = {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12};

  constexpr size_t kRhsInputSize = 12;
  float rhs_input[kRhsInputSize];
  std::iota(std::begin(rhs_input), std::end(rhs_input), 7);

  constexpr float kExpect[] = {74.,  80.,  86.,  92.,  173., 188., 203., 218.,
                               272., 296., 320., 344., 371., 404., 437., 470.};
  constexpr int kOutputDims[] = {3, 2, 2, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      true,   // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kLhsInput,
                                        rhs_input, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2) {
  constexpr int kLhsInputDims[] = {4, 2, 1, 3, 2};
  constexpr int kRhsInputDims[] = {3, 3, 2, 4};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr size_t kLhsInputSize = 12;
  float lhs_input[kLhsInputSize];
  std::iota(std::begin(lhs_input), std::end(lhs_input), 1);

  constexpr size_t kRhsInputSize = 24;
  float rhs_input[kRhsInputSize];
  std::iota(std::begin(rhs_input), std::end(rhs_input), 7);

  constexpr float kExpect[] = {
      29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101., 112., 123., 134.,
      53.,  56.,  59.,  62.,  121., 128., 135., 142., 189., 200., 211., 222.,
      77.,  80.,  83.,  86.,  177., 184., 191., 198., 277., 288., 299., 310.,
      137., 152., 167., 182., 173., 192., 211., 230., 209., 232., 255., 278.,
      257., 272., 287., 302., 325., 344., 363., 382., 393., 416., 439., 462.,
      377., 392., 407., 422., 477., 496., 515., 534., 577., 600., 623., 646.};
  constexpr int kOutputDims[] = {4, 2, 3, 3, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, lhs_input,
                                        rhs_input, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2LHSAdjoint) {
  constexpr int kLhsInputDims[] = {4, 2, 1, 2, 3};
  constexpr int kRhsInputDims[] = {3, 3, 2, 4};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr float kLhsInput[] = {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};

  constexpr size_t kRhsInputSize = 24;
  float rhs_input[kRhsInputSize];
  std::iota(std::begin(rhs_input), std::end(rhs_input), 7);

  constexpr float kExpect[] = {
      29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101., 112., 123., 134.,
      53.,  56.,  59.,  62.,  121., 128., 135., 142., 189., 200., 211., 222.,
      77.,  80.,  83.,  86.,  177., 184., 191., 198., 277., 288., 299., 310.,
      137., 152., 167., 182., 173., 192., 211., 230., 209., 232., 255., 278.,
      257., 272., 287., 302., 325., 344., 363., 382., 393., 416., 439., 462.,
      377., 392., 407., 422., 477., 496., 515., 534., 577., 600., 623., 646.};
  constexpr int kOutputDims[] = {4, 2, 3, 3, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      true,   // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kLhsInput,
                                        rhs_input, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2RHSAdjoint) {
  constexpr int kLhsInputDims[] = {4, 2, 1, 3, 2};
  constexpr int kRhsInputDims[] = {3, 3, 4, 2};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr size_t kLhsInputSize = 12;
  float lhs_input[kLhsInputSize];
  std::iota(std::begin(lhs_input), std::end(lhs_input), 1);

  constexpr float kRhsInput[] = {7,  11, 8,  12, 9,  13, 10, 14,
                                 15, 19, 16, 20, 17, 21, 18, 22,
                                 23, 27, 24, 28, 25, 29, 26, 30};

  constexpr float kExpect[] = {
      29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101., 112., 123., 134.,
      53.,  56.,  59.,  62.,  121., 128., 135., 142., 189., 200., 211., 222.,
      77.,  80.,  83.,  86.,  177., 184., 191., 198., 277., 288., 299., 310.,
      137., 152., 167., 182., 173., 192., 211., 230., 209., 232., 255., 278.,
      257., 272., 287., 302., 325., 344., 363., 382., 393., 416., 439., 462.,
      377., 392., 407., 422., 477., 496., 515., 534., 577., 600., 623., 646.};
  constexpr int kOutputDims[] = {4, 2, 3, 3, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      true,   // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, lhs_input,
                                        kRhsInput, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2BothAdjoint) {
  constexpr int kLhsInputDims[] = {4, 2, 1, 2, 3};
  constexpr int kRhsInputDims[] = {3, 3, 4, 2};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr float kLhsInput[] = {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};

  constexpr float kRhsInput[] = {7,  11, 8,  12, 9,  13, 10, 14,
                                 15, 19, 16, 20, 17, 21, 18, 22,
                                 23, 27, 24, 28, 25, 29, 26, 30};

  constexpr float kExpect[] = {
      29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101., 112., 123., 134.,
      53.,  56.,  59.,  62.,  121., 128., 135., 142., 189., 200., 211., 222.,
      77.,  80.,  83.,  86.,  177., 184., 191., 198., 277., 288., 299., 310.,
      137., 152., 167., 182., 173., 192., 211., 230., 209., 232., 255., 278.,
      257., 272., 287., 302., 325., 344., 363., 382., 393., 416., 439., 462.,
      377., 392., 407., 422., 477., 496., 515., 534., 577., 600., 623., 646.};
  constexpr int kOutputDims[] = {4, 2, 3, 3, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      true,  // adj_x
      true,  // adj_y
      false  // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kLhsInput,
                                        kRhsInput, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_BroadcastFromRHS) {
  constexpr int kLhsInputDims[] = {2, 4, 5};
  constexpr int kRhsInputDims[] = {4, 3, 1, 5, 2};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr size_t kLhsInputSize = 20;
  float lhs_input[kLhsInputSize];
  std::iota(std::begin(lhs_input), std::end(lhs_input), 1);

  constexpr size_t kRhsInputSize = 30;
  float rhs_input[kRhsInputSize];
  std::iota(std::begin(rhs_input), std::end(rhs_input), 7);

  constexpr float kExpect[] = {185.,  200.,  460.,  500.,  735.,  800.,
                               1010., 1100., 335.,  350.,  860.,  900.,
                               1385., 1450., 1910., 2000., 485.,  500.,
                               1260., 1300., 2035., 2100., 2810., 2900.};
  constexpr int kOutputDims[] = {4, 3, 1, 4, 2};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, lhs_input,
                                        rhs_input, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(ConstRHSBatchMatMulOpModelRHSNotAdjoint) {
  constexpr int kLhsInputDims[] = {3, 1, 6, 2};
  constexpr int kRhsInputDims[] = {2, 2, 3};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr float kLhsInput[] = {6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7};

  constexpr float kRhsInput[] = {6, 3, 7, 4, 6, 9};

  constexpr float kExpect[] = {48, 36, 69, 58, 45, 85, 72, 72, 123,
                               36, 42, 68, 58, 45, 85, 46, 51, 84};
  constexpr int kOutputDims[] = {3, 1, 6, 3};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kLhsInput,
                                        kRhsInput, kOutputDims, kExpect,
                                        output_data, true);
  // Eval twice to make sure constant transposed RHS is persistent.
  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kLhsInput,
                                        kRhsInput, kOutputDims, kExpect,
                                        output_data, true, false);
}

TF_LITE_MICRO_TEST(QuantizedBatchMatMulOpTestSimpleTestQuantizedInt8) {
  constexpr int kLhsInputDims[] = {2, 2, 10};
  constexpr int kRhsInputDims[] = {2, 10, 3};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr float kLhsInput[] = {
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  };
  constexpr int kLhsInputCount = std::extent<decltype(kLhsInput)>::value;

  constexpr float kRhsInput[] = {
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  };
  constexpr int kRhsInputCount = std::extent<decltype(kRhsInput)>::value;

  constexpr int8_t kExpect[] = {22, 22, 22, 56, 56, 56};
  constexpr int kOutputDims[] = {2, 2, 3};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  constexpr float output_data[kOutputCount] = {23, 23, 23, 57, 57, 57};

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestQuantizationParams<int8_t, kLhsInputCount>
      quantization_params_lhs = {0.0f,    // scale
                                 0,       // zero_point
                                 -63.5f,  // data_min
                                 64.0f,   // data_max
                                 {}};
  tflite::testing::TestQuantizationParams<int8_t, kRhsInputCount>
      quantization_params_rhs = {0.0f,    // scale
                                 0,       // zero_point
                                 -63.5f,  // data_min
                                 64.0f,   // data_max
                                 {}};
  tflite::testing::TestQuantizationParams<int8_t, kOutputCount>
      quantization_params_output = {0.0f,     // scale
                                    0,        // zero_point
                                    -127.0f,  // data_min
                                    128.0f,   // data_max
                                    {}};

  tflite::testing::TestBatchMatMulQuantized<int8_t>(
      params, &quantization_params_lhs, &quantization_params_rhs,
      &quantization_params_output, kInputDims, kLhsInput, kRhsInput,
      kOutputDims, kExpect, output_data);
}

TF_LITE_MICRO_TEST(QuantizedBatchMatMulOpTestSimpleTestQuantizedInt16) {
  constexpr int kLhsInputDims[] = {2, 2, 10};
  constexpr int kRhsInputDims[] = {2, 10, 3};
  const int* kInputDims[tflite::testing::kNumInputs] = {kLhsInputDims,
                                                        kRhsInputDims};

  constexpr float kLhsInput[] = {
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  };
  constexpr int kLhsInputCount = std::extent<decltype(kLhsInput)>::value;

  constexpr float kRhsInput[] = {
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  };
  constexpr int kRhsInputCount = std::extent<decltype(kRhsInput)>::value;

  constexpr int16_t kExpect[] = {23, 23, 23, 57, 57, 57};
  constexpr int kOutputDims[] = {2, 2, 3};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  constexpr float output_data[kOutputCount] = {23, 23, 23, 57, 57, 57};

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestQuantizationParams<int16_t, kLhsInputCount>
      quantization_params_lhs = {};
  quantization_params_lhs.scale = 10.0f / std::numeric_limits<int16_t>::max();
  tflite::testing::TestQuantizationParams<int16_t, kRhsInputCount>
      quantization_params_rhs = {};
  quantization_params_rhs.scale = 10.0f / std::numeric_limits<int16_t>::max();

  tflite::testing::TestQuantizationParams<int16_t, kOutputCount>
      quantization_params_output = {};
  quantization_params_output.scale = 1.0f;

  tflite::testing::TestBatchMatMulQuantized<int16_t>(
      params, &quantization_params_lhs, &quantization_params_rhs,
      &quantization_params_output, kInputDims, kLhsInput, kRhsInput,
      kOutputDims, kExpect, output_data);
}

TF_LITE_MICRO_TESTS_END
