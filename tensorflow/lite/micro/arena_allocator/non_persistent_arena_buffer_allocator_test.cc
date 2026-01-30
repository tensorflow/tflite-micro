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

#include "tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.h"

#include <cstdint>

#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

// Test the creation of the resizable buffer and exercise resize.
TEST(NonPersistentArenaBufferAllocatorTest, TestResizableBuffer) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::NonPersistentArenaBufferAllocator allocator(arena, arena_size);

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(10, 1);
  EXPECT_EQ(resizable_buf, arena);

  EXPECT_EQ(kTfLiteOk, allocator.ResizeBuffer(resizable_buf, /*size=*/100,
                                              /*alignment=*/1));
  EXPECT_EQ(static_cast<size_t>(100), allocator.GetNonPersistentUsedBytes());

  EXPECT_EQ(kTfLiteOk, allocator.ResizeBuffer(resizable_buf, /*size=*/10,
                                              /*alignment=*/1));
  EXPECT_EQ(static_cast<size_t>(10), allocator.GetNonPersistentUsedBytes());

  EXPECT_EQ(
      allocator.ResizeBuffer(resizable_buf, /*size=*/1000, /*alignment=*/1),
      kTfLiteOk);
  EXPECT_EQ(static_cast<size_t>(1000), allocator.GetNonPersistentUsedBytes());
}

// Test allocate and deallocate temp buffer.
TEST(NonPersistentArenaBufferAllocatorTest, TestTempBuffer) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::NonPersistentArenaBufferAllocator allocator(arena, arena_size);

  constexpr size_t allocation_size = 100;
  uint8_t* temp = allocator.AllocateTemp(/*size=*/allocation_size,
                                         /*alignment=*/1);
  EXPECT_EQ(allocation_size, allocator.GetNonPersistentUsedBytes());

  EXPECT_EQ(allocator.GetAvailableMemory(/*alignment=*/1),
            arena_size - allocation_size);

  // Reset temp allocations and ensure GetAvailableMemory() is back to the
  // starting size:
  allocator.DeallocateTemp(temp);
  allocator.ResetTempAllocations();

  EXPECT_EQ(static_cast<size_t>(0), allocator.GetNonPersistentUsedBytes());
  EXPECT_EQ(allocator.GetAvailableMemory(/*alignment=*/1), arena_size);
}

// Resizable buffer cannot be allocated if there is still outstanding temp
// buffers.
TEST(NonPersistentArenaBufferAllocatorTest,
     TestAllocateResizeFailIfTempStillExists) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::NonPersistentArenaBufferAllocator allocator(arena, arena_size);

  constexpr size_t allocation_size = 100;
  uint8_t* temp = allocator.AllocateTemp(/*size=*/allocation_size,
                                         /*alignment=*/1);
  // Deallocate does not free up temp buffer.
  allocator.DeallocateTemp(temp);

  EXPECT_EQ(allocator.AllocateResizableBuffer(allocation_size, 1), nullptr);
}

// Resizable buffer can be allocated if there are no  outstanding temp buffers.
TEST(NonPersistentArenaBufferAllocatorTest, TestAllocateResizePassIfNoTemp) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::NonPersistentArenaBufferAllocator allocator(arena, arena_size);

  constexpr size_t allocation_size = 100;
  uint8_t* temp = allocator.AllocateTemp(/*size=*/allocation_size,
                                         /*alignment=*/1);
  // Deallocate does not free up temp buffer.
  allocator.DeallocateTemp(temp);
  EXPECT_EQ(allocator.ResetTempAllocations(), kTfLiteOk);

  EXPECT_EQ(allocator.AllocateResizableBuffer(allocation_size, 1), arena);
}

// Cannot allocate more than one resizable buffer.
TEST(NonPersistentArenaBufferAllocatorTest,
     TestAllocateResizableFailIfResizableExists) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::NonPersistentArenaBufferAllocator allocator(arena, arena_size);

  constexpr size_t allocation_size = 100;
  EXPECT_TRUE(allocator.AllocateResizableBuffer(/*size=*/allocation_size,
                                                /*alignment=*/1) != nullptr);

  EXPECT_TRUE(allocator.AllocateResizableBuffer(/*size=*/allocation_size,
                                                /*alignment=*/1) == nullptr);
}

// ResetTempAllocations() fail if there are still outstanding temp buffers
TEST(NonPersistentArenaBufferAllocatorTest,
     TestResetTempFailIfTempStillExists) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::NonPersistentArenaBufferAllocator allocator(arena, arena_size);

  constexpr size_t allocation_size = 100;
  allocator.AllocateTemp(/*size=*/allocation_size,
                         /*alignment=*/1);

  EXPECT_EQ(allocator.ResetTempAllocations(), kTfLiteError);
}

// Request more than allocated size for temp will fail
TEST(NonPersistentArenaBufferAllocatorTest,
     TestAllocateTempFailIfExceedAllowance) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::NonPersistentArenaBufferAllocator allocator(arena, arena_size);

  EXPECT_TRUE(allocator.AllocateTemp(/*size=*/arena_size + 1,
                                     /*alignment=*/1) == nullptr);
}

// Request more than allocated size for resizable will fail
TEST(NonPersistentArenaBufferAllocatorTest,
     TestAllocateTempFailIfExceedAllowance2) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::NonPersistentArenaBufferAllocator allocator(arena, arena_size);

  EXPECT_TRUE(allocator.AllocateResizableBuffer(
                  /*size=*/arena_size + 1, /*alignment=*/1) == nullptr);

  constexpr size_t allocation_size = 100;
  uint8_t* resizable_buffer =
      allocator.AllocateResizableBuffer(/*size=*/allocation_size,
                                        /*alignment=*/1);
  EXPECT_EQ(resizable_buffer, arena);

  EXPECT_EQ(allocator.ResizeBuffer(resizable_buffer, /*size=*/arena_size + 1,
                                   /*alignment=*/1),
            kTfLiteError);
}

// GetNonPersistentUsedBytes() reports memory for both resizable buffer and temp
// buffers.
TEST(NonPersistentArenaBufferAllocatorTest, TestGetNonPersistentUsedBytes) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::NonPersistentArenaBufferAllocator allocator(arena, arena_size);

  constexpr size_t allocation_size = 100;
  EXPECT_TRUE(arena ==
              allocator.AllocateResizableBuffer(/*size=*/allocation_size,
                                                /*alignment=*/1));

  EXPECT_TRUE(allocator.AllocateTemp(/*size=*/arena_size - allocation_size,
                                     /*alignment=*/1) != nullptr);

  EXPECT_EQ(allocator.GetNonPersistentUsedBytes(), arena_size);
}

TF_LITE_MICRO_TESTS_MAIN
