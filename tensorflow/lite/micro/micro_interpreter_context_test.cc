/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/micro_interpreter_context.h"

#include <cstdint>

#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_interpreter_graph.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

using ::tflite::testing::IntArrayFromInts;

namespace tflite {
namespace {

tflite::MicroInterpreterContext CreateMicroInterpreterContext() {
  // Some targets do not support dynamic memory (i.e., no malloc or new), thus,
  // the test need to place non-transient memories in static variables. This is
  // safe because tests are guaranteed to run serially.
  constexpr size_t kArenaSize = 1024;
  static uint8_t tensor_arena[kArenaSize];

  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  MicroAllocator* micro_allocator =
      MicroAllocator::Create(tensor_arena, kArenaSize);
  static MicroInterpreterGraph micro_graph(nullptr, nullptr, nullptr, nullptr);

  tflite::MicroInterpreterContext micro_context(micro_allocator, model,
                                                &micro_graph);
  return micro_context;
}

// Test structure for external context payload.
struct TestExternalContextPayloadData {
  // Opaque blob
  alignas(4) uint8_t blob_data[128];
};
}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// Ensures that a regular set and get pair works ok.
TF_LITE_MICRO_TEST(TestSetGetExternalContextSuccess) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();
  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kInvoke);

  tflite::TestExternalContextPayloadData payload;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          micro_context.set_external_context(&payload));

  tflite::TestExternalContextPayloadData* returned_external_context =
      reinterpret_cast<tflite::TestExternalContextPayloadData*>(
          micro_context.external_context());

  // What is returned should be the same as what is set.
  TF_LITE_MICRO_EXPECT((void*)returned_external_context == (void*)(&payload));
}

TF_LITE_MICRO_TEST(TestGetExternalContextWithoutSetShouldReturnNull) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();

  tflite::TestExternalContextPayloadData* returned_external_context =
      reinterpret_cast<tflite::TestExternalContextPayloadData*>(
          micro_context.external_context());

  // Return a null if nothing is set before.
  TF_LITE_MICRO_EXPECT((void*)returned_external_context == (nullptr));
}

TF_LITE_MICRO_TEST(TestSetExternalContextCanOnlyBeCalledOnce) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();
  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kPrepare);
  tflite::TestExternalContextPayloadData payload;

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          micro_context.set_external_context(&payload));

  // Another set should fail.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          micro_context.set_external_context(&payload));
}

TF_LITE_MICRO_TEST(TestSetExternalContextToNullShouldFail) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();
  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kPrepare);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          micro_context.set_external_context(nullptr));
}

TF_LITE_MICRO_TEST(TestGetTempInputTensor) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();

  TfLiteNode node;
  int input_data[] = {2, 0, 1};
  node.inputs = IntArrayFromInts(input_data);

  TfLiteTensor* input1 = micro_context.AllocateTempInputTensor(&node, 0);
  TF_LITE_MICRO_EXPECT_TRUE(input1 != nullptr);
  micro_context.DeallocateTempTfLiteTensor(input1);

  TfLiteTensor* input2 = micro_context.AllocateTempInputTensor(&node, 1);
  TF_LITE_MICRO_EXPECT_TRUE(input2 != nullptr);
  micro_context.DeallocateTempTfLiteTensor(input2);

  TfLiteTensor* invalid_input = micro_context.AllocateTempInputTensor(&node, 2);
  TF_LITE_MICRO_EXPECT_TRUE(invalid_input == nullptr);
}

TF_LITE_MICRO_TEST(TestGetTempOutputTensor) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();

  TfLiteNode node;
  int output_data[] = {1, 0};
  node.outputs = IntArrayFromInts(output_data);

  TfLiteTensor* output = micro_context.AllocateTempOutputTensor(&node, 0);
  TF_LITE_MICRO_EXPECT_TRUE(output != nullptr);
  micro_context.DeallocateTempTfLiteTensor(output);

  TfLiteTensor* invalid_output =
      micro_context.AllocateTempOutputTensor(&node, 1);
  TF_LITE_MICRO_EXPECT_TRUE(invalid_output == nullptr);
}

TF_LITE_MICRO_TEST(TestAllocateTempBuffer) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();
  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kPrepare);
  uint8_t* buffer1 =
      micro_context.AllocateTempBuffer(10, tflite::MicroArenaBufferAlignment());
  TF_LITE_MICRO_EXPECT(buffer1 != nullptr);
}

TF_LITE_MICRO_TEST(TestGetTempIntermediateTensor) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();

  TfLiteNode node;
  int intermediate_data[] = {1, 0};
  node.intermediates = IntArrayFromInts(intermediate_data);

  TfLiteTensor* output = micro_context.AllocateTempIntermediateTensor(&node, 0);
  TF_LITE_MICRO_EXPECT_TRUE(output != nullptr);
  micro_context.DeallocateTempTfLiteTensor(output);

  TfLiteTensor* invalid_output =
      micro_context.AllocateTempIntermediateTensor(&node, 1);
  TF_LITE_MICRO_EXPECT_TRUE(invalid_output == nullptr);
}

TF_LITE_MICRO_TESTS_END
