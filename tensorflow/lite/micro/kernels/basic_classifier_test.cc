/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <cstring>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

void TestBasicClassifier(
    int* input_dims_data, const int32_t* input_data,
    int* target_indices_dims_data, const int32_t* target_indices_data,
    int* thresholds_dims_data, const int32_t* thresholds_data,
    int* detected_dims_data, bool* detected_data,
    int* target_posteriors_dims_data, int32_t* target_posteriors_data,
    const bool* golden_detected, const int32_t* golden_target_posteriors) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* target_indices_dims =
      IntArrayFromInts(target_indices_dims_data);
  TfLiteIntArray* thresholds_dims = IntArrayFromInts(thresholds_dims_data);
  TfLiteIntArray* detected_dims = IntArrayFromInts(detected_dims_data);
  TfLiteIntArray* target_posteriors_dims =
      IntArrayFromInts(target_posteriors_dims_data);
  const int output_len = ElementCount(*detected_dims);
  constexpr int kInputsSize = 3;
  constexpr int kOutputsSize = 2;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(target_indices_data, target_indices_dims),
      CreateTensor(thresholds_data, thresholds_dims),
      CreateTensor(detected_data, detected_dims),
      CreateTensor(target_posteriors_data, target_posteriors_dims),
  };
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {2, 3, 4};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_BASIC_CLASSIFIER();
  micro::KernelRunner runner(registration, tensors, kTensorsSize, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_len; ++i) {
    EXPECT_EQ(golden_detected[i], detected_data[i]);
    EXPECT_EQ(golden_target_posteriors[i], target_posteriors_data[i]);
  }
}

void TestBasicClassifierInvalidTargetIndex(
    int* input_dims_data, const int32_t* input_data,
    int* target_indices_dims_data, const int32_t* target_indices_data,
    int* thresholds_dims_data, const int32_t* thresholds_data,
    int* detected_dims_data, bool* detected_data,
    int* target_posteriors_dims_data, int32_t* target_posteriors_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* target_indices_dims =
      IntArrayFromInts(target_indices_dims_data);
  TfLiteIntArray* thresholds_dims = IntArrayFromInts(thresholds_dims_data);
  TfLiteIntArray* detected_dims = IntArrayFromInts(detected_dims_data);
  TfLiteIntArray* target_posteriors_dims =
      IntArrayFromInts(target_posteriors_dims_data);
  constexpr int kInputsSize = 3;
  constexpr int kOutputsSize = 2;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(target_indices_data, target_indices_dims),
      CreateTensor(thresholds_data, thresholds_dims),
      CreateTensor(detected_data, detected_dims),
      CreateTensor(target_posteriors_data, target_posteriors_dims),
  };
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {2, 3, 4};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_BASIC_CLASSIFIER();
  micro::KernelRunner runner(registration, tensors, kTensorsSize, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_NE(kTfLiteOk, runner.Invoke());
}

void TestBasicClassifierMismatchedDims(
    int* input_dims_data, const int32_t* input_data,
    int* target_indices_dims_data, const int32_t* target_indices_data,
    int* thresholds_dims_data, const int32_t* thresholds_data,
    int* detected_dims_data, bool* detected_data,
    int* target_posteriors_dims_data, int32_t* target_posteriors_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* target_indices_dims =
      IntArrayFromInts(target_indices_dims_data);
  TfLiteIntArray* thresholds_dims = IntArrayFromInts(thresholds_dims_data);
  TfLiteIntArray* detected_dims = IntArrayFromInts(detected_dims_data);
  TfLiteIntArray* target_posteriors_dims =
      IntArrayFromInts(target_posteriors_dims_data);
  constexpr int kInputsSize = 3;
  constexpr int kOutputsSize = 2;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(target_indices_data, target_indices_dims),
      CreateTensor(thresholds_data, thresholds_dims),
      CreateTensor(detected_data, detected_dims),
      CreateTensor(target_posteriors_data, target_posteriors_dims),
  };
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {2, 3, 4};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_BASIC_CLASSIFIER();
  micro::KernelRunner runner(registration, tensors, kTensorsSize, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_NE(kTfLiteOk, runner.InitAndPrepare());
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(BasicClassifierTest, TrueCase) {
  int input_shape[] = {1, 5};
  int target_indices_shape[] = {1, 1};
  int thresholds_shape[] = {1, 1};
  int detected_shape[] = {1, 1};
  int target_posteriors_shape[] = {1, 1};
  const int32_t input[] = {15, 15, 15, 18, 15};
  const int32_t target_indices[] = {3};
  const int32_t thresholds[] = {17};
  bool detected[1];
  int32_t target_posteriors[1];
  const bool golden_detected[] = {true};
  const int32_t golden_target_posteriors[] = {18};
  memset(detected, 0, sizeof(detected));
  memset(target_posteriors, 0, sizeof(target_posteriors));
  tflite::testing::TestBasicClassifier(
      input_shape, input, target_indices_shape, target_indices,
      thresholds_shape, thresholds, detected_shape, detected,
      target_posteriors_shape, target_posteriors, golden_detected,
      golden_target_posteriors);
}

TEST(BasicClassifierTest, FalseCase) {
  int input_shape[] = {1, 2};
  int target_indices_shape[] = {1, 1};
  int thresholds_shape[] = {1, 1};
  int detected_shape[] = {1, 1};
  int target_posteriors_shape[] = {1, 1};
  const int32_t input[] = {0, 12345};
  const int32_t target_indices[] = {0};
  const int32_t thresholds[] = {12345};
  bool detected[1];
  int32_t target_posteriors[1];
  const bool golden_detected[] = {false};
  const int32_t golden_target_posteriors[] = {0};
  memset(detected, 0, sizeof(detected));
  memset(target_posteriors, 0, sizeof(target_posteriors));
  tflite::testing::TestBasicClassifier(
      input_shape, input, target_indices_shape, target_indices,
      thresholds_shape, thresholds, detected_shape, detected,
      target_posteriors_shape, target_posteriors, golden_detected,
      golden_target_posteriors);
}

TEST(BasicClassifierTest, MulticlassCase) {
  int input_shape[] = {1, 5};
  int target_indices_shape[] = {1, 2};
  int thresholds_shape[] = {1, 2};
  int detected_shape[] = {1, 2};
  int target_posteriors_shape[] = {1, 2};
  const int32_t input[] = {14, 654, 321, 865, 653};
  const int32_t target_indices[] = {2, 4};
  const int32_t thresholds[] = {322, 653};
  bool detected[2];
  int32_t target_posteriors[2];
  const bool golden_detected[] = {false, true};
  const int32_t golden_target_posteriors[] = {321, 653};
  memset(detected, 0, sizeof(detected));
  memset(target_posteriors, 0, sizeof(target_posteriors));
  tflite::testing::TestBasicClassifier(
      input_shape, input, target_indices_shape, target_indices,
      thresholds_shape, thresholds, detected_shape, detected,
      target_posteriors_shape, target_posteriors, golden_detected,
      golden_target_posteriors);
}

TEST(BasicClassifierTest, InvalidTargetIndex) {
  int input_shape[] = {1, 3};
  int target_indices_shape[] = {1, 1};
  int thresholds_shape[] = {1, 1};
  int detected_shape[] = {1, 1};
  int target_posteriors_shape[] = {1, 1};
  const int32_t input[] = {10, 20, 30};
  const int32_t target_indices[] = {5};
  const int32_t thresholds[] = {15};
  bool detected[1];
  int32_t target_posteriors[1];

  tflite::testing::TestBasicClassifierInvalidTargetIndex(
      input_shape, input, target_indices_shape, target_indices,
      thresholds_shape, thresholds, detected_shape, detected,
      target_posteriors_shape, target_posteriors);
}

TEST(BasicClassifierTest, MismatchedDims) {
  int input_shape[] = {1, 5};
  int target_indices_shape[] = {1, 2};
  int thresholds_shape[] = {1, 3};
  int detected_shape[] = {1, 2};
  int target_posteriors_shape[] = {1, 2};
  const int32_t input[] = {10, 20, 30, 40, 50};
  const int32_t target_indices[] = {1, 2};
  const int32_t thresholds[] = {15, 25, 35};
  bool detected[2];
  int32_t target_posteriors[2];

  tflite::testing::TestBasicClassifierMismatchedDims(
      input_shape, input, target_indices_shape, target_indices,
      thresholds_shape, thresholds, detected_shape, detected,
      target_posteriors_shape, target_posteriors);
}

TF_LITE_MICRO_TESTS_MAIN
