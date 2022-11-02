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

#include "tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.h"

#include <cstdint>

#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

// Test that the right amount of memory are allocated.
TF_LITE_MICRO_TEST(TestGetPersistentUsedBytes) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::PersistentArenaBufferAllocator allocator(arena, arena_size);

  const size_t size1 = 10;
  allocator.AllocatePersistentBuffer(size1, 1);
  TF_LITE_MICRO_EXPECT_EQ(size1, allocator.GetPersistentUsedBytes());

  const size_t size2 = 15;
  allocator.AllocatePersistentBuffer(size2, 1);

  TF_LITE_MICRO_EXPECT_EQ(size1 + size2, allocator.GetPersistentUsedBytes());
}

// Test allocation shall fail if total memory exceeds the limit.
TF_LITE_MICRO_TEST(TestAllocatePersistBufferShallFailIfExceedLimit) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::PersistentArenaBufferAllocator allocator(arena, arena_size);

  const size_t size1 = 10;
  uint8_t* persist1 = allocator.AllocatePersistentBuffer(size1, 1);
  TF_LITE_MICRO_EXPECT(persist1 != nullptr);

  const size_t size2 = arena_size - size1 + 1;
  uint8_t* persist2 = allocator.AllocatePersistentBuffer(size2, 1);

  TF_LITE_MICRO_EXPECT(persist2 == nullptr);
}

// Test allocation shall pass if total memory does not exceed the limit.
TF_LITE_MICRO_TEST(TestAllocatePersistBufferShallPassIfWithinLimit) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::PersistentArenaBufferAllocator allocator(arena, arena_size);

  const size_t size1 = 10;
  uint8_t* persist1 = allocator.AllocatePersistentBuffer(size1, 1);
  TF_LITE_MICRO_EXPECT(persist1 != nullptr);

  const size_t size2 = arena_size - size1;
  uint8_t* persist2 = allocator.AllocatePersistentBuffer(size2, 1);

  TF_LITE_MICRO_EXPECT(persist2 != nullptr);
  TF_LITE_MICRO_EXPECT_EQ(arena_size, allocator.GetPersistentUsedBytes());
}

// Test alignment works.
TF_LITE_MICRO_TEST(TestAllocatePersistBufferAligns) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::PersistentArenaBufferAllocator allocator(arena, arena_size);

  const size_t size1 = 10;
  const size_t alignment = 16;
  uint8_t* persist1 = allocator.AllocatePersistentBuffer(size1, alignment);
  TF_LITE_MICRO_EXPECT(persist1 != nullptr);
  TF_LITE_MICRO_EXPECT_EQ(
      (reinterpret_cast<std::uintptr_t>(persist1)) % alignment,
      static_cast<std::uintptr_t>(0));
  TF_LITE_MICRO_EXPECT_GE(allocator.GetPersistentUsedBytes(), size1);

  const size_t size2 = 16;
  uint8_t* persist2 = allocator.AllocatePersistentBuffer(size2, alignment);
  TF_LITE_MICRO_EXPECT(persist2 != nullptr);
  TF_LITE_MICRO_EXPECT_EQ(
      (reinterpret_cast<std::uintptr_t>(persist2)) % alignment,
      static_cast<std::uintptr_t>(0));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(persist1 - persist2), size2);
  TF_LITE_MICRO_EXPECT_GE(allocator.GetPersistentUsedBytes(), size1);
}
TF_LITE_MICRO_TESTS_END
