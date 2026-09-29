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

#include <cstddef>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

template <typename InputT, typename OutputT, size_t N>
void TestCast(const InputT (&input)[N], const OutputT (&golden)[N]) {
  OutputT output_data[N];

  int dims_data[] = {1, static_cast<int>(N)};
  TfLiteIntArray* dims = IntArrayFromInts(dims_data);

  constexpr int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input, dims),
      CreateTensor(output_data, dims),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_CAST();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(golden[i], output_data[i]);
  }
}

template <typename IntT>
void TestFloatToInt(float pos_overflow, float neg_overflow) {
  [[maybe_unused]] constexpr bool is_signed =
      std::numeric_limits<IntT>::is_signed;
  const float input[] = {100.f,
                         1.0f,
                         0.f,
                         0.4f,
                         1.999f,
                         1.1f,
                         -1.0f,
                         -100.f,
                         pos_overflow,
                         neg_overflow,
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::quiet_NaN()};
  const IntT golden[] = {100,
                         1,
                         0,
                         0,
                         1,
                         1,
                         is_signed ? static_cast<IntT>(-1) : IntT{0},
                         is_signed ? static_cast<IntT>(-100) : IntT{0},
                         std::numeric_limits<IntT>::max(),
                         std::numeric_limits<IntT>::min(),
                         std::numeric_limits<IntT>::max(),
                         std::numeric_limits<IntT>::min(),
                         0};
  TestCast(input, golden);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(CastTest, CastFloatToInt8) {
  tflite::testing::TestFloatToInt<int8_t>(200.f, -200.f);
}

TEST(CastTest, CastFloatToInt16) {
  tflite::testing::TestFloatToInt<int16_t>(40000.f, -40000.f);
}

TEST(CastTest, CastFloatToInt32) {
  tflite::testing::TestFloatToInt<int32_t>(1e15f, -1e15f);
}

TEST(CastTest, CastFloatToUInt32) {
  tflite::testing::TestFloatToInt<uint32_t>(1e15f, -1e15f);
}

TEST(CastTest, CastFloatToBool) {
  const float input[] = {1.0f,
                         0.0f,
                         -1.0f,
                         0.001f,
                         std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::quiet_NaN()};
  const bool golden[] = {true, false, true, true, true, true};
  tflite::testing::TestCast(input, golden);
}

TEST(CastTest, CastInt8ToFloat) {
  const int8_t input[] = {123, 0, 1, 2, 3, 4};
  const float golden[] = {123.f, 0.f, 1.f, 2.f, 3.f, 4.f};
  tflite::testing::TestCast(input, golden);
}

TEST(CastTest, CastInt16ToFloat) {
  const int16_t input[] = {123, 0, 1, 2, 3, 4};
  const float golden[] = {123.f, 0.f, 1.f, 2.f, 3.f, 4.f};
  tflite::testing::TestCast(input, golden);
}

TEST(CastTest, CastInt16ToInt32) {
  const int16_t input[] = {123, 0, 1, 2, 3, 4};
  const int32_t golden[] = {123, 0, 1, 2, 3, 4};
  tflite::testing::TestCast(input, golden);
}

TEST(CastTest, CastInt32ToInt16) {
  const int32_t input[] = {123, 0, 1, 2, 3, 4};
  const int16_t golden[] = {123, 0, 1, 2, 3, 4};
  tflite::testing::TestCast(input, golden);
}

TEST(CastTest, CastUInt32ToInt32) {
  const uint32_t input[] = {100, 200, 300, 400, 500, 600};
  const int32_t golden[] = {100, 200, 300, 400, 500, 600};
  tflite::testing::TestCast(input, golden);
}

TEST(CastTest, CastInt32ToUInt32) {
  const int32_t input[] = {100, 200, 300, 400, 500, 600};
  const uint32_t golden[] = {100, 200, 300, 400, 500, 600};
  tflite::testing::TestCast(input, golden);
}

TEST(CastTest, CastBoolToFloat) {
  const bool input[] = {true, true, false, true, false, true};
  const float golden[] = {1.f, 1.0f, 0.f, 1.0f, 0.0f, 1.0f};
  tflite::testing::TestCast(input, golden);
}

TF_LITE_MICRO_TESTS_MAIN
