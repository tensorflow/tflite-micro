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

#include <cstdint>
#include <limits>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr float kTestTolerance = 1e-05;
constexpr int kNumInputs = 3;
constexpr int kNumOutputs = 1;
constexpr int kInputTensorIndex = 0;
constexpr int kBlockShapeTensorIndex = 1;
constexpr int kCropTensorIndex = 2;
constexpr int kOutputTensorIndex = 3;

// min/max are used to compute scale, zero-point (asymmetric)
template <typename T, size_t kInputSize, size_t kOutputSize>
struct TestQuantParams {
  // quantization parameters
  float data_min;              // input data minimum value
  float data_max;              // input data maximum value
  T output_data[kOutputSize];  // quantized output storage
  T input_data[kInputSize];    // quantized input storage
};

TfLiteStatus ExecuteBatchToSpaceNdTest(TfLiteTensor* tensors,
                                       int tensors_count) {
  int kInputArrayData[] = {kNumInputs, kInputTensorIndex,
                           kBlockShapeTensorIndex, kCropTensorIndex};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  int kOutputArrayData[] = {kNumOutputs, kOutputTensorIndex};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TFLMRegistration registration = tflite::Register_BATCH_TO_SPACE_ND();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, nullptr);

  TfLiteStatus status = runner.InitAndPrepare();
  if (status != kTfLiteOk) {
    return status;
  }
  status = runner.Invoke();

  return status;
}

template <typename T>
TfLiteStatus TestBatchToSpaceNd(int* input_dims_data[kNumInputs],
                                const T* input_data,
                                const int32_t* block_shape_data,
                                const int32_t* crop_data, int* output_dims_data,
                                const T* golden_data, T* output_data) {
  TfLiteIntArray* input_dims =
      IntArrayFromInts(input_dims_data[kInputTensorIndex]);
  TfLiteIntArray* block_shape_dims =
      IntArrayFromInts(input_dims_data[kBlockShapeTensorIndex]);
  TfLiteIntArray* crop_dims =
      IntArrayFromInts(input_dims_data[kCropTensorIndex]);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int kTensorsCount = kNumInputs + kNumOutputs;
  TfLiteTensor tensors[kTensorsCount];
  tensors[kInputTensorIndex] =
      tflite::testing::CreateTensor(input_data, input_dims);
  tensors[kBlockShapeTensorIndex] =
      tflite::testing::CreateTensor(block_shape_data, block_shape_dims);
  tensors[kBlockShapeTensorIndex].allocation_type = kTfLiteMmapRo;
  tensors[kCropTensorIndex] =
      tflite::testing::CreateTensor(crop_data, crop_dims);
  tensors[kCropTensorIndex].allocation_type = kTfLiteMmapRo;
  tensors[kOutputTensorIndex] =
      tflite::testing::CreateTensor(output_data, output_dims);

  TfLiteStatus status = ExecuteBatchToSpaceNdTest(tensors, kTensorsCount);
  if (status != kTfLiteOk) {
    return status;
  }

  // check output dimensions (relocated) against original dimensions
  TF_LITE_MICRO_EXPECT_EQ(output_dims->size,
                          tensors[kOutputTensorIndex].dims->size);
  TF_LITE_MICRO_CHECK_FAIL();
  for (int i = 0; i < output_dims->size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(output_dims->data[i],
                            tensors[kOutputTensorIndex].dims->data[i]);
    // TODO(b/158102673): workaround for not having fatal test assertions.
    TF_LITE_MICRO_CHECK_FAIL();
  }

  // check output data against expected
  const int output_count = ElementCount(*output_dims);
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(golden_data[i], output_data[i], kTestTolerance);
    // TODO(b/158102673): workaround for not having fatal test assertions.
    TF_LITE_MICRO_CHECK_FAIL();
  }

  return kTfLiteOk;
}

template <typename T, size_t kInCount, size_t kOutCount>
TfLiteStatus TestBatchToSpaceNdQuantized(
    TestQuantParams<T, kInCount, kOutCount>& params,
    int* input_dims_data[kNumInputs], const float* input_data,
    const int32_t* block_shape_data, const int32_t* crop_data,
    int* output_dims_data, const float* golden_data) {
  TfLiteIntArray* input_dims =
      IntArrayFromInts(input_dims_data[kInputTensorIndex]);
  TfLiteIntArray* block_shape_dims =
      IntArrayFromInts(input_dims_data[kBlockShapeTensorIndex]);
  TfLiteIntArray* crop_dims =
      IntArrayFromInts(input_dims_data[kCropTensorIndex]);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr float kMaxMultiplier =
      std::numeric_limits<T>::max() /
      static_cast<float>(std::numeric_limits<T>::max() + 1);
  int zero_point = tflite::testing::ZeroPointFromMinMax<T>(
      params.data_min, params.data_max * kMaxMultiplier);
  float scale = tflite::testing::ScaleFromMinMax<T>(
      params.data_min, params.data_max * kMaxMultiplier);

  constexpr int kTensorsCount = kNumInputs + kNumOutputs;
  TfLiteTensor tensors[kTensorsCount];
  tensors[kInputTensorIndex] = tflite::testing::CreateQuantizedTensor(
      input_data, params.input_data, input_dims, scale, zero_point);
  tensors[kBlockShapeTensorIndex] =
      tflite::testing::CreateTensor(block_shape_data, block_shape_dims);
  tensors[kBlockShapeTensorIndex].allocation_type = kTfLiteMmapRo;
  tensors[kCropTensorIndex] =
      tflite::testing::CreateTensor(crop_data, crop_dims);
  tensors[kCropTensorIndex].allocation_type = kTfLiteMmapRo;
  tensors[kOutputTensorIndex] = tflite::testing::CreateQuantizedTensor(
      params.output_data, output_dims, scale, zero_point);

  TfLiteStatus status = ExecuteBatchToSpaceNdTest(tensors, kTensorsCount);
  if (status != kTfLiteOk) {
    return status;
  }

  // check output dimensions (relocated) against original dimensions
  TF_LITE_MICRO_EXPECT_EQ(output_dims->size,
                          tensors[kOutputTensorIndex].dims->size);
  TF_LITE_MICRO_CHECK_FAIL();
  for (int i = 0; i < output_dims->size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(output_dims->data[i],
                            tensors[kOutputTensorIndex].dims->data[i]);
    // TODO(b/158102673): workaround for not having fatal test assertions.
    TF_LITE_MICRO_CHECK_FAIL();
  }

  // check output data against expected
  const int output_count = ElementCount(*output_dims);
  const float quantization_tolerance =
      (params.data_max - params.data_min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  for (int i = 0; i < output_count; i++) {
    float output_dequantized_data =
        (params.output_data[i] - zero_point) * scale;
    TF_LITE_MICRO_EXPECT_NEAR(golden_data[i], output_dequantized_data,
                              quantization_tolerance);
    // TODO(b/158102673): workaround for not having fatal test assertions.
    TF_LITE_MICRO_CHECK_FAIL();
  }

  return kTfLiteOk;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestInvalidOutputShapeTest) {
  int kInputDims[] = {3, 2, 4, 1};
  int kBlockShapeDims[] = {1, 1};
  int kCropDims[] = {2, 1, 2};
  int kOutputDims[] = {3, 1, 1, 1};  // invalid shape

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int32_t kBlockShape[] = {2};
  constexpr int32_t kCrop[] = {0, 0};
  constexpr float kGolden[] = {0};  // placeholder data, not used
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;
  float output_data[kOutputCount];

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          tflite::testing::TestBatchToSpaceNd(
                              kInputDimsArray, kInput, kBlockShape, kCrop,
                              kOutputDims, kGolden, output_data));
}

TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestValidOutputShapeTest) {
  int kInputDims[] = {3, 2, 4, 1};
  int kBlockShapeDims[] = {1, 1};
  int kCropDims[] = {2, 1, 2};
  int kOutputDims[] = {3, 1, 8, 1};

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int32_t kBlockShape[] = {2};
  constexpr int32_t kCrop[] = {0, 0};
  constexpr float kGolden[] = {1, 5, 2, 6, 3, 7, 4, 8};
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;
  float output_data[kOutputCount];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestBatchToSpaceNd(
                     kInputDimsArray, kInput, kBlockShape, kCrop, kOutputDims,
                     kGolden, output_data));
}

TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestSimpleConstTest) {
  int kInputDims[] = {4, 4, 2, 2, 1};
  int kBlockShapeDims[] = {1, 2};
  int kCropDims[] = {2, 2, 2};
  int kOutputDims[] = {4, 1, 4, 4, 1};

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr float kInput[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  };
  constexpr int32_t kBlockShape[] = {2, 2};
  constexpr int32_t kCrop[] = {0, 0, 0, 0};
  constexpr float kGolden[] = {
      1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16,
  };
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;
  float output_data[kOutputCount];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestBatchToSpaceNd(
                     kInputDimsArray, kInput, kBlockShape, kCrop, kOutputDims,
                     kGolden, output_data));
}

// non-quantized test
TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestSimpleConstTestInt8) {
  int kInputDims[] = {4, 4, 2, 2, 1};
  int kBlockShapeDims[] = {1, 2};
  int kCropDims[] = {2, 2, 2};
  int kOutputDims[] = {4, 1, 4, 4, 1};

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr int8_t kInput[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  };
  constexpr int32_t kBlockShape[] = {2, 2};
  constexpr int32_t kCrop[] = {0, 0, 0, 0};
  constexpr int8_t kGolden[] = {
      1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16,
  };
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;
  int8_t output_data[kOutputCount];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestBatchToSpaceNd(
                     kInputDimsArray, kInput, kBlockShape, kCrop, kOutputDims,
                     kGolden, output_data));
}

TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestBatchOneConstTest) {
  int kInputDims[] = {4, 1, 2, 2, 1};
  int kBlockShapeDims[] = {1, 2};
  int kCropDims[] = {2, 2, 2};
  int kOutputDims[] = {4, 1, 2, 2, 1};

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr float kInput[] = {1, 2, 3, 4};
  constexpr int32_t kBlockShape[] = {1, 1};
  constexpr int32_t kCrop[] = {0, 0, 0, 0};
  constexpr float kGolden[] = {1, 2, 3, 4};
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;
  float output_data[kOutputCount];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestBatchToSpaceNd(
                     kInputDimsArray, kInput, kBlockShape, kCrop, kOutputDims,
                     kGolden, output_data));
}

// non-quantized test
TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestSimpleConstTestInt8EmptyOutput) {
  int kInputDims[] = {4, 4, 2, 2, 1};
  int kBlockShapeDims[] = {1, 2};
  int kCropDims[] = {2, 2, 2};
  int kOutputDims[] = {4, 1, 4, 0, 1};

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr int8_t kInput[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  };
  constexpr int32_t kBlockShape[] = {2, 2};
  constexpr int32_t kCrop[] = {0, 0, 2, 2};
  constexpr int8_t kGolden[] = {0};  // placeholder data, not used
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;
  int8_t output_data[kOutputCount];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestBatchToSpaceNd(
                     kInputDimsArray, kInput, kBlockShape, kCrop, kOutputDims,
                     kGolden, output_data));
}

TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestInvalidShapeTest) {
  int kInputDims[] = {4, 3, 2, 2, 1};
  int kBlockShapeDims[] = {1, 2};
  int kCropDims[] = {2, 2, 2};
  int kOutputDims[] = {4, 0, 0, 0, 0};  // placeholder dims, not used

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr float kInput[] = {0};  // placeholder data, not used
  constexpr int32_t kBlockShape[] = {2, 2};
  constexpr int32_t kCrop[] = {0, 0, 0, 0};
  constexpr float kGolden[] = {0};  // placeholder data, not used
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;
  float output_data[kOutputCount];

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          tflite::testing::TestBatchToSpaceNd(
                              kInputDimsArray, kInput, kBlockShape, kCrop,
                              kOutputDims, kGolden, output_data));
}

TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestInvalidCropsConstTest) {
  int kInputDims[] = {4, 3, 2, 2, 1};
  int kBlockShapeDims[] = {1, 2};
  int kCropDims[] = {2, 2, 2};
  int kOutputDims[] = {4, 0, 0, 0, 0};  // placeholder dims, not used

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr float kInput[] = {0};  // placeholder data, not used
  constexpr int32_t kBlockShape[] = {2, 2};
  constexpr int32_t kCrop[] = {0, 0, 0, -1};
  constexpr float kGolden[] = {0};  // placeholder data, not used
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;
  float output_data[kOutputCount];

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          tflite::testing::TestBatchToSpaceNd(
                              kInputDimsArray, kInput, kBlockShape, kCrop,
                              kOutputDims, kGolden, output_data));
}

TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestSimple3DConstTest) {
  int kInputDims[] = {3, 4, 4, 1};
  int kBlockShapeDims[] = {1, 1};
  int kCropDims[] = {2, 1, 2};
  int kOutputDims[] = {3, 2, 8, 1};

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr float kInput[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  };
  constexpr int32_t kBlockShape[] = {2};
  constexpr int32_t kCrop[] = {0, 0};
  constexpr float kGolden[] = {
      1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16,
  };
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;
  float output_data[kOutputCount];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestBatchToSpaceNd(
                     kInputDimsArray, kInput, kBlockShape, kCrop, kOutputDims,
                     kGolden, output_data));
}

TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestSimple3DConstTestWithCrops) {
  int kInputDims[] = {3, 4, 4, 1};
  int kBlockShapeDims[] = {1, 1};
  int kCropDims[] = {2, 1, 2};
  int kOutputDims[] = {3, 2, 6, 1};

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr float kInput[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  };
  constexpr int32_t kBlockShape[] = {2};
  constexpr int32_t kCrop[] = {1, 1};
  constexpr float kGolden[] = {9, 2, 10, 3, 11, 4, 13, 6, 14, 7, 15, 8};
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;
  float output_data[kOutputCount];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestBatchToSpaceNd(
                     kInputDimsArray, kInput, kBlockShape, kCrop, kOutputDims,
                     kGolden, output_data));
}

TF_LITE_MICRO_TEST(BatchToSpaceNDOpTestSimple3DConstTestWithCropsINT8) {
  int kInputDims[] = {3, 4, 4, 1};
  int kBlockShapeDims[] = {1, 1};
  int kCropDims[] = {2, 1, 2};
  int kOutputDims[] = {3, 2, 6, 1};

  int* kInputDimsArray[tflite::testing::kNumInputs];
  kInputDimsArray[tflite::testing::kInputTensorIndex] = kInputDims;
  kInputDimsArray[tflite::testing::kBlockShapeTensorIndex] = kBlockShapeDims;
  kInputDimsArray[tflite::testing::kCropTensorIndex] = kCropDims;

  constexpr float kInput[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  };
  constexpr int kInputCount = std::extent<decltype(kInput)>::value;
  constexpr int32_t kBlockShape[] = {2};
  constexpr int32_t kCrop[] = {1, 1};
  constexpr float kGolden[] = {9, 2, 10, 3, 11, 4, 13, 6, 14, 7, 15, 8};
  constexpr int kOutputCount = std::extent<decltype(kGolden)>::value;

  tflite::testing::TestQuantParams<int8_t, kInputCount, kOutputCount> params = {
      -16, 16, {}, {}};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestBatchToSpaceNdQuantized(
                     params, kInputDimsArray, kInput, kBlockShape, kCrop,
                     kOutputDims, kGolden));
}

TF_LITE_MICRO_TESTS_END
