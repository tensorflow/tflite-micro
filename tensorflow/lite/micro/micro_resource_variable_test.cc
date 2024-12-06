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

#include "tensorflow/lite/micro/micro_resource_variable.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {

constexpr int kMaxBufferSize = 1024;
uint8_t buffer_[kMaxBufferSize];
int last_allocation_size_;

void* AllocateMockBuffer(TfLiteContext* context, size_t size) {
  last_allocation_size_ = size;
  return buffer_;
}

TfLiteContext* GetMockContext() {
  static TfLiteContext mock_context = {};
  mock_context.AllocatePersistentBuffer = AllocateMockBuffer;
  return &mock_context;
}

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(CreateVariables) {
  tflite::MicroResourceVariables* resource_variables =
      tflite::MicroResourceVariables::Create(
          tflite::MicroAllocator::Create(tflite::buffer_,
                                         tflite::kMaxBufferSize),
          4);
  int id1 = resource_variables->CreateIdIfNoneFound("", "var1");
  TF_LITE_MICRO_EXPECT_GE(id1, 0);

  int id2 = resource_variables->CreateIdIfNoneFound("", "var2");
  TF_LITE_MICRO_EXPECT_NE(id1, id2);

  int id3 = resource_variables->CreateIdIfNoneFound("foo", "var1");
  TF_LITE_MICRO_EXPECT_NE(id1, id3);
  TF_LITE_MICRO_EXPECT_NE(id2, id3);

  int id4 = resource_variables->CreateIdIfNoneFound("foo", "var2");
  TF_LITE_MICRO_EXPECT_NE(id1, id4);
  TF_LITE_MICRO_EXPECT_NE(id2, id4);
  TF_LITE_MICRO_EXPECT_NE(id3, id4);

  TF_LITE_MICRO_EXPECT_EQ(id2,
                          resource_variables->CreateIdIfNoneFound("", "var2"));
  TF_LITE_MICRO_EXPECT_EQ(id1,
                          resource_variables->CreateIdIfNoneFound("", "var1"));
  TF_LITE_MICRO_EXPECT_EQ(
      id4, resource_variables->CreateIdIfNoneFound("foo", "var2"));
  TF_LITE_MICRO_EXPECT_EQ(
      id3, resource_variables->CreateIdIfNoneFound("foo", "var1"));
}

TF_LITE_MICRO_TEST(AllocateResourceBuffers) {
  tflite::MicroResourceVariables* resource_variables =
      tflite::MicroResourceVariables::Create(
          tflite::MicroAllocator::Create(tflite::buffer_,
                                         tflite::kMaxBufferSize),
          2);
  int id1 = resource_variables->CreateIdIfNoneFound("", "var1");
  TF_LITE_MICRO_EXPECT_GE(id1, 0);

  int id2 = resource_variables->CreateIdIfNoneFound("", "var2");
  TF_LITE_MICRO_EXPECT_NE(id1, id2);

  TfLiteTensor tensor = {};
  tensor.bytes = 42;
  resource_variables->Allocate(id1, tflite::GetMockContext(), &tensor);
  TF_LITE_MICRO_EXPECT_EQ(42, tflite::last_allocation_size_);

  tensor.bytes = 100;
  resource_variables->Allocate(id2, tflite::GetMockContext(), &tensor);
  TF_LITE_MICRO_EXPECT_EQ(100, tflite::last_allocation_size_);
}

TF_LITE_MICRO_TEST(VerifyAssignAndReadResourceBuffer) {
  tflite::MicroResourceVariables* resource_variables =
      tflite::MicroResourceVariables::Create(
          tflite::MicroAllocator::Create(tflite::buffer_,
                                         tflite::kMaxBufferSize),
          1);
  int id = resource_variables->CreateIdIfNoneFound("", "var1");
  TF_LITE_MICRO_EXPECT_GE(id, 0);

  TfLiteTensor tensor = {};
  const int bytes = 32 * sizeof(int32_t);
  tensor.bytes = bytes;
  resource_variables->Allocate(id, tflite::GetMockContext(), &tensor);
  TF_LITE_MICRO_EXPECT_EQ(bytes, tflite::last_allocation_size_);

  int32_t golden[32] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  int dims[] = {1, 32};
  TfLiteEvalTensor assign_tensor = {
      .data = {golden},
      .dims = tflite::testing::IntArrayFromInts(dims),

      .type = kTfLiteFloat32,
  };
  resource_variables->Assign(
      id, tflite::EvalTensorBytes(&assign_tensor),
      tflite::micro::GetTensorData<void>(&assign_tensor));

  int32_t buffer[32];
  TfLiteEvalTensor read_tensor = {
      .data = {buffer},
      .dims = tflite::testing::IntArrayFromInts(dims),
      .type = kTfLiteInt32,
  };
  resource_variables->Read(id, &read_tensor);
  for (int i = 0; i < 32; i++) {
    TF_LITE_MICRO_EXPECT_EQ(buffer[i], golden[i]);
  }
}

TF_LITE_MICRO_TEST(CreateVariablesNullContainer) {
  tflite::MicroResourceVariables* resource_variables =
      tflite::MicroResourceVariables::Create(
          tflite::MicroAllocator::Create(tflite::buffer_,
                                         tflite::kMaxBufferSize),
          4);
  int id1 = resource_variables->CreateIdIfNoneFound(nullptr, "var1");
  TF_LITE_MICRO_EXPECT_GE(id1, 0);

  int id2 = resource_variables->CreateIdIfNoneFound(nullptr, "var2");
  TF_LITE_MICRO_EXPECT_NE(id1, id2);

  TF_LITE_MICRO_EXPECT_EQ(
      id2, resource_variables->CreateIdIfNoneFound(nullptr, "var2"));
  TF_LITE_MICRO_EXPECT_EQ(
      id1, resource_variables->CreateIdIfNoneFound(nullptr, "var1"));
}

TF_LITE_MICRO_TESTS_END
