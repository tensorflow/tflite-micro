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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {
using ::tflite::testing::CreateTensor;
using ::tflite::testing::IntArrayFromInts;

// The layout of tensors is fixed.
constexpr int kDimsIndex = 0;
constexpr int kValueIndex = 1;
constexpr int kOutputIndex = 2;
constexpr int kInputsTensor[] = {2, kDimsIndex, kValueIndex};
constexpr int kOutputsTensor[] = {1, kOutputIndex};

// This function is NOT thread safe.
template <typename DimsType, typename ValueType, typename OutputType>
tflite::micro::KernelRunner CreateFillTestRunner(
    int* dims_shape, DimsType* dims_data, int* value_shape,
    ValueType* value_data, int* output_shape, OutputType* output_data) {
  // Some targets do not support dynamic memory (i.e., no malloc or new), thus,
  // the test need to place non-transitent memories in static variables. This is
  // safe because tests are guaranteed to run serially.
  // Both below structures are trivially destructible.
  static TFLMRegistration registration;
  static TfLiteTensor tensors[3];

  tensors[0] = CreateTensor(dims_data, IntArrayFromInts(dims_shape));
  if (dims_data != nullptr) {
    // dims must be a const tensor
    tensors[0].allocation_type = kTfLiteMmapRo;
  }
  tensors[1] = CreateTensor(value_data, IntArrayFromInts(value_shape));
  tensors[2] = CreateTensor(output_data, IntArrayFromInts(output_shape));

  // The output type matches the value type.
  TF_LITE_MICRO_EXPECT_EQ(tensors[kOutputIndex].type,
                          tensors[kValueIndex].type);

  registration = tflite::Register_FILL();
  tflite::micro::KernelRunner runner = tflite::micro::KernelRunner(
      registration, tensors, sizeof(tensors) / sizeof(TfLiteTensor),
      IntArrayFromInts(const_cast<int*>(kInputsTensor)),
      IntArrayFromInts(const_cast<int*>(kOutputsTensor)),
      /*builtin_data=*/nullptr);
  return runner;
}

template <typename DimsType, typename ValueType, typename OutputType>
void TestFill(int* dims_shape, DimsType* dims_data, int* value_shape,
              ValueType* value_data, int* output_shape,
              OutputType* output_data) {
  tflite::micro::KernelRunner runner =
      CreateFillTestRunner(dims_shape, dims_data, value_shape, value_data,
                           output_shape, output_data);

  TF_LITE_MICRO_EXPECT_EQ(runner.InitAndPrepare(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(runner.Invoke(), kTfLiteOk);

  // The output shape must match the shape requested via dims.
  const auto output_rank = output_shape[0];
  if (dims_data != nullptr) {
    const auto requested_rank = dims_shape[1];  // yes, 1
    if (output_rank == requested_rank) {
      for (int i = 0; i < requested_rank; ++i) {
        TF_LITE_MICRO_EXPECT_EQ(output_shape[i + 1], dims_data[i]);
      }
    } else {
      TF_LITE_MICRO_FAIL(
          "output shape does not match shape requested via dims");
    }
  }

  // The output elements contain the fill value.
  const auto elements = tflite::ElementCount(*IntArrayFromInts(output_shape));
  for (int i = 0; i < elements; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(output_data[i], value_data[0]);
  }
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FillFloatInt64Dims) {
  constexpr int kDim1 = 2;
  constexpr int kDim2 = 2;
  constexpr int kDim3 = 2;

  int dims_shape[] = {1, 3};
  int64_t dims_data[] = {kDim1, kDim2, kDim3};

  int value_shape[] = {0};
  float value_data[] = {4.0};

  int output_shape[] = {3, kDim1, kDim2, kDim3};
  float output_data[kDim1 * kDim2 * kDim3];

  TestFill(dims_shape, dims_data, value_shape, value_data, output_shape,
           output_data);
}

// Fill a 2x2x2 tensor with a int32 scalar value. The dimension of the tensor is
// of int64 type.
TF_LITE_MICRO_TEST(FillInt32Int64Dims) {
  constexpr int kDim1 = 2;
  constexpr int kDim2 = 2;
  constexpr int kDim3 = 2;

  int dims_shape[] = {1, 3};
  int64_t dims_data[] = {kDim1, kDim2, kDim3};

  int value_shape[] = {0};
  int32_t value_data[] = {4};

  int output_shape[] = {3, kDim1, kDim2, kDim3};
  int32_t output_data[kDim1 * kDim2 * kDim3];

  TestFill(dims_shape, dims_data, value_shape, value_data, output_shape,
           output_data);
}

// Fill a 2x2x2 tensor with a int8 scalar value. The dimension of the tensor is
// of int32 type.
TF_LITE_MICRO_TEST(FillInt8Int32Dims) {
  constexpr int kDim1 = 2;
  constexpr int kDim2 = 2;
  constexpr int kDim3 = 2;

  int dims_shape[] = {1, 3};
  int32_t dims_data[] = {kDim1, kDim2, kDim3};

  int value_shape[] = {0};
  int8_t value_data[] = {4};

  int output_shape[] = {3, kDim1, kDim2, kDim3};
  int8_t output_data[kDim1 * kDim2 * kDim3];

  TestFill(dims_shape, dims_data, value_shape, value_data, output_shape,
           output_data);
}

TF_LITE_MICRO_TEST(FillInt8NonConstDimsTensorFail) {
  constexpr int kDim1 = 2;
  constexpr int kDim2 = 2;
  constexpr int kDim3 = 2;

  // Simulate the dims tensor with dynamic data. Note that shape is always
  // known.
  int dims_shape[] = {1, 3};
  int32_t* dims_data = nullptr;

  int value_shape[] = {0};
  int8_t value_data[] = {4};

  int output_shape[] = {3, kDim1, kDim2, kDim3};
  int8_t output_data[kDim1 * kDim2 * kDim3];

  tflite::micro::KernelRunner runner =
      CreateFillTestRunner(dims_shape, dims_data, value_shape, value_data,
                           output_shape, output_data);

  TF_LITE_MICRO_EXPECT_EQ(runner.InitAndPrepare(), kTfLiteError);
}

TF_LITE_MICRO_TEST(FillFloatInt32Dims) {
  constexpr int kDim1 = 2;
  constexpr int kDim2 = 2;
  constexpr int kDim3 = 2;

  int dims_shape[] = {1, 3};
  int32_t dims_data[] = {kDim1, kDim2, kDim3};

  int value_shape[] = {0};
  float value_data[] = {4.0};

  int output_shape[] = {3, kDim1, kDim2, kDim3};
  float output_data[kDim1 * kDim2 * kDim3];

  TestFill(dims_shape, dims_data, value_shape, value_data, output_shape,
           output_data);
}

TF_LITE_MICRO_TEST(FillScalar) {
  int dims_shape[] = {1, 0};
  int64_t dims_data[] = {0};

  int value_shape[] = {0};
  float value_data[] = {4.0};

  int output_shape[] = {0};
  float output_data[] = {0};

  TestFill(dims_shape, dims_data, value_shape, value_data, output_shape,
           output_data);
}

// When input dimension tensor mismatch with the output tensor's dimension,
// the FILL op shall return error at init/prepare stage.
TF_LITE_MICRO_TEST(FillInputDimsMismatchWithOutputShallFail) {
  constexpr int kDim1 = 2;
  constexpr int kDim2 = 2;
  constexpr int kDim3 = 2;

  int dims_shape[] = {1, 3};
  int64_t dims_data[] = {kDim1, kDim2, kDim3};

  int value_shape[] = {0};
  int8_t value_data[] = {4};

  // Output shape is supposed to be the same as dims_data.
  // Intentionally +1 to kDim1 to verify the code catches this error.
  int output_shape[] = {3, kDim1 + 1, kDim2, kDim3};
  int8_t output_data[(kDim1 + 1) * kDim2 * kDim3];

  tflite::micro::KernelRunner runner =
      CreateFillTestRunner(dims_shape, dims_data, value_shape, value_data,
                           output_shape, output_data);

  TF_LITE_MICRO_EXPECT_EQ(runner.InitAndPrepare(), kTfLiteError);
}

TF_LITE_MICRO_TESTS_END
