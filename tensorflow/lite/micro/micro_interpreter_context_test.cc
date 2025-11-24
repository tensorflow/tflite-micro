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
  alignas(16) static uint8_t tensor_arena[kArenaSize];

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

// Ensures that a regular set and get pair works ok during state kInvoke.
TF_LITE_MICRO_TEST(TestSetGetExternalContextSuccessInvoke) {
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
  TF_LITE_MICRO_EXPECT(returned_external_context == &payload);
}

// Ensures that a regular set and get pair works ok during state kInit.
TF_LITE_MICRO_TEST(TestSetGetExternalContextSuccessInit) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();
  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kInit);

  tflite::TestExternalContextPayloadData payload;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          micro_context.set_external_context(&payload));

  tflite::TestExternalContextPayloadData* returned_external_context =
      reinterpret_cast<tflite::TestExternalContextPayloadData*>(
          micro_context.external_context());

  // What is returned should be the same as what is set.
  TF_LITE_MICRO_EXPECT(returned_external_context == &payload);
}

TF_LITE_MICRO_TEST(TestGetExternalContextWithoutSetShouldReturnNull) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();

  void* returned_external_context = micro_context.external_context();

  // Return a null if nothing is set before.
  TF_LITE_MICRO_EXPECT(returned_external_context == nullptr);
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

  // Null set should fail.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          micro_context.set_external_context(nullptr));
  tflite::TestExternalContextPayloadData* returned_external_context =
      reinterpret_cast<tflite::TestExternalContextPayloadData*>(
          micro_context.external_context());
  // Payload should be unchanged.
  TF_LITE_MICRO_EXPECT(&payload == returned_external_context);
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

TF_LITE_MICRO_TEST(TestSetDecompressionMemory) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();

  constexpr size_t kAltMemorySize = 1;
  alignas(16) uint8_t g_alt_memory[kAltMemorySize];
  std::initializer_list<tflite::MicroContext::AlternateMemoryRegion>
      alt_memory_region = {{g_alt_memory, kAltMemorySize}};
  TfLiteStatus status;

  // Test that all of the MicroInterpreterContext fences are correct, by
  // forcing the MicroInterpreterContext state. The SetDecompressionMemory
  // method should only be allowed during the kInit state, and can only be
  // set once.  This is because alternate decompression memory is allocated
  // during the application initiated kPrepare state, and all memory has already
  // been statically allocted during the kInvoke state.

  // fail during Prepare state
  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kPrepare);
  status = micro_context.SetDecompressionMemory(alt_memory_region);
  TF_LITE_MICRO_EXPECT(status == kTfLiteError);

  // fail during Invoke state
  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kInvoke);
  status = micro_context.SetDecompressionMemory(alt_memory_region);
  TF_LITE_MICRO_EXPECT(status == kTfLiteError);

  // succeed during Init state
  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kInit);
  status = micro_context.SetDecompressionMemory(alt_memory_region);
  TF_LITE_MICRO_EXPECT(status == kTfLiteOk);

  // fail on second Init state attempt
  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kInit);
  status = micro_context.SetDecompressionMemory(alt_memory_region);
  TF_LITE_MICRO_EXPECT(status == kTfLiteError);
}

TF_LITE_MICRO_TEST(TestAllocateDecompressionMemory) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();

  constexpr size_t kAltMemorySize = 30;
  constexpr size_t kAllocateSize = 10;
  alignas(16) uint8_t g_alt_memory[kAltMemorySize];
  std::initializer_list<tflite::MicroContext::AlternateMemoryRegion>
      alt_memory_region = {{g_alt_memory, kAltMemorySize}};

  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kInit);
  TfLiteStatus status = micro_context.SetDecompressionMemory(alt_memory_region);
  TF_LITE_MICRO_EXPECT(status == kTfLiteOk);

  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kPrepare);

  // allocate first 10 bytes at offset 0 (total allocated is 10 bytes)
  uint8_t* p = static_cast<uint8_t*>(micro_context.AllocateDecompressionMemory(
      kAllocateSize, tflite::MicroArenaBufferAlignment()));
  TF_LITE_MICRO_EXPECT(p == &g_alt_memory[0]);

  // allocate next 10 bytes at offset 16 (total allocated is 26 bytes)
  p = static_cast<uint8_t*>(micro_context.AllocateDecompressionMemory(
      kAllocateSize, tflite::MicroArenaBufferAlignment()));
  TF_LITE_MICRO_EXPECT(p == &g_alt_memory[tflite::MicroArenaBufferAlignment()]);

  // fail next allocation of 10 bytes (offset 32 > available memory)
  p = static_cast<uint8_t*>(micro_context.AllocateDecompressionMemory(
      kAllocateSize, tflite::MicroArenaBufferAlignment()));
  TF_LITE_MICRO_EXPECT(p == nullptr);
}

TF_LITE_MICRO_TEST(TestResetDecompressionMemory) {
  tflite::MicroInterpreterContext micro_context =
      tflite::CreateMicroInterpreterContext();

  constexpr size_t kAltMemorySize = 30;
  constexpr size_t kAllocateSize = 10;
  alignas(16) uint8_t g_alt_memory[kAltMemorySize];
  std::initializer_list<tflite::MicroContext::AlternateMemoryRegion>
      alt_memory_region = {{g_alt_memory, kAltMemorySize}};

  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kInit);
  TfLiteStatus status = micro_context.SetDecompressionMemory(alt_memory_region);
  TF_LITE_MICRO_EXPECT(status == kTfLiteOk);

  micro_context.SetInterpreterState(
      tflite::MicroInterpreterContext::InterpreterState::kPrepare);

  // allocate first 10 bytes at offset 0 (total allocated is 10 bytes)
  uint8_t* p = static_cast<uint8_t*>(micro_context.AllocateDecompressionMemory(
      kAllocateSize, tflite::MicroArenaBufferAlignment()));
  TF_LITE_MICRO_EXPECT(p == &g_alt_memory[0]);

  // allocate next 10 bytes at offset 16 (total allocated is 26 bytes)
  p = static_cast<uint8_t*>(micro_context.AllocateDecompressionMemory(
      kAllocateSize, tflite::MicroArenaBufferAlignment()));
  TF_LITE_MICRO_EXPECT(p == &g_alt_memory[tflite::MicroArenaBufferAlignment()]);

  micro_context.ResetDecompressionMemoryAllocations();

  // allocate first 10 bytes again at offset 0 (total allocated is 10 bytes)
  p = static_cast<uint8_t*>(micro_context.AllocateDecompressionMemory(
      kAllocateSize, tflite::MicroArenaBufferAlignment()));
  TF_LITE_MICRO_EXPECT(p == &g_alt_memory[0]);
}

TF_LITE_MICRO_TESTS_END
