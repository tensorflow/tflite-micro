/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/fake_micro_context.h"

#include <cstdint>

#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/mock_micro_graph.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {
using ::tflite::testing::CreateTensor;
using ::tflite::testing::IntArrayFromInts;

tflite::FakeMicroContext CreateFakeMicroContext(
    SingleArenaBufferAllocator* simple_memory_allocator,
    MicroGraph* micro_graph) {
  // Some targets do not support dynamic memory (i.e., no malloc or new), thus,
  // the test need to place non-transitent memories in static variables. This is
  // safe because tests are guarateed to run serially.
  // Below structures are trivially destructible.
  static TfLiteTensor tensors[2];
  static int input_shape[] = {1, 3};
  static int input_data[] = {1, 2, 3};

  static int output_shape[] = {1, 3};
  static float output_data[3];

  tensors[0] = CreateTensor(input_data, IntArrayFromInts(input_shape));
  tensors[1] = CreateTensor(output_data, IntArrayFromInts(output_shape));

  tflite::FakeMicroContext fake_micro_context(tensors, simple_memory_allocator,
                                              micro_graph);
  return fake_micro_context;
}

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestGetBeforeRequestScratchBufferWouldReturnNull) {
  constexpr size_t kArenaSize = 1024;
  uint8_t arena_buffer[kArenaSize];
  tflite::SingleArenaBufferAllocator simple_memory_allocator(arena_buffer,
                                                             kArenaSize);
  tflite::MockMicroGraph dummy_micro_graph{&simple_memory_allocator};

  tflite::FakeMicroContext micro_context = tflite::CreateFakeMicroContext(
      &simple_memory_allocator, &dummy_micro_graph);

  TF_LITE_MICRO_EXPECT(micro_context.GetScratchBuffer(0) == nullptr);
}

TF_LITE_MICRO_TEST(TestRequestScratchBufferAndThenGetShouldSucceed) {
  constexpr size_t kArenaSize = 1024;
  uint8_t arena_buffer[kArenaSize];
  tflite::SingleArenaBufferAllocator simple_memory_allocator(arena_buffer,
                                                             kArenaSize);
  tflite::MockMicroGraph dummy_micro_graph{&simple_memory_allocator};

  tflite::FakeMicroContext micro_context = tflite::CreateFakeMicroContext(
      &simple_memory_allocator, &dummy_micro_graph);

  constexpr size_t kScratchBufferSize = 16;
  int scratch_buffer_index = -1;
  TF_LITE_MICRO_EXPECT_EQ(micro_context.RequestScratchBufferInArena(
                              kScratchBufferSize, &scratch_buffer_index),
                          kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(scratch_buffer_index, 0);
  TF_LITE_MICRO_EXPECT(micro_context.GetScratchBuffer(scratch_buffer_index) !=
                       nullptr);

  TF_LITE_MICRO_EXPECT_EQ(micro_context.RequestScratchBufferInArena(
                              kScratchBufferSize, &scratch_buffer_index),
                          kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(scratch_buffer_index, 1);
  TF_LITE_MICRO_EXPECT(micro_context.GetScratchBuffer(scratch_buffer_index) !=
                       nullptr);
}

TF_LITE_MICRO_TESTS_END
