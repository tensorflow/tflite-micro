/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.h"

#include <cstdint>

#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

TEST(RecordingSingleArenaBufferAllocatorTest, TestRecordsTailAllocations) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::RecordingSingleArenaBufferAllocator allocator(arena, arena_size);

  uint8_t* result =
      allocator.AllocatePersistentBuffer(/*size=*/10, /*alignment=*/1);
  EXPECT_NE(result, nullptr);
  EXPECT_EQ(allocator.GetUsedBytes(), static_cast<size_t>(10));
  EXPECT_EQ(allocator.GetRequestedBytes(), static_cast<size_t>(10));
  EXPECT_EQ(allocator.GetAllocatedCount(), static_cast<size_t>(1));

  result = allocator.AllocatePersistentBuffer(/*size=*/20, /*alignment=*/1);
  EXPECT_NE(result, nullptr);
  EXPECT_EQ(allocator.GetUsedBytes(), static_cast<size_t>(30));
  EXPECT_EQ(allocator.GetRequestedBytes(), static_cast<size_t>(30));
  EXPECT_EQ(allocator.GetAllocatedCount(), static_cast<size_t>(2));
}

TEST(RecordingSingleArenaBufferAllocatorTest,
     TestRecordsMisalignedTailAllocations) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::RecordingSingleArenaBufferAllocator allocator(arena, arena_size);

  uint8_t* result =
      allocator.AllocatePersistentBuffer(/*size=*/10, /*alignment=*/12);
  EXPECT_NE(result, nullptr);
  // Validate used bytes in 8 byte range that can included alignment of 12:
  EXPECT_GE(allocator.GetUsedBytes(), static_cast<size_t>(10));
  EXPECT_LE(allocator.GetUsedBytes(), static_cast<size_t>(20));
  EXPECT_EQ(allocator.GetRequestedBytes(), static_cast<size_t>(10));
  EXPECT_EQ(allocator.GetAllocatedCount(), static_cast<size_t>(1));
}

TEST(RecordingSingleArenaBufferAllocatorTest,
     TestDoesNotRecordFailedTailAllocations) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::RecordingSingleArenaBufferAllocator allocator(arena, arena_size);

  uint8_t* result =
      allocator.AllocatePersistentBuffer(/*size=*/2048, /*alignment=*/1);
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(allocator.GetUsedBytes(), static_cast<size_t>(0));
  EXPECT_EQ(allocator.GetRequestedBytes(), static_cast<size_t>(0));
  EXPECT_EQ(allocator.GetAllocatedCount(), static_cast<size_t>(0));
}

TEST(RecordingSingleArenaBufferAllocatorTest, TestRecordsHeadSizeAdjustment) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::RecordingSingleArenaBufferAllocator allocator(arena, arena_size);

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, 1);
  EXPECT_NE(resizable_buf, nullptr);

  EXPECT_EQ(kTfLiteOk,
            allocator.ResizeBuffer(resizable_buf, /*size=*/5, /*alignment=*/1));
  EXPECT_EQ(allocator.GetUsedBytes(), static_cast<size_t>(5));
  EXPECT_EQ(allocator.GetRequestedBytes(), static_cast<size_t>(5));
  // Head adjustments do not count as an allocation:
  EXPECT_EQ(allocator.GetAllocatedCount(), static_cast<size_t>(0));

  uint8_t* result =
      allocator.AllocatePersistentBuffer(/*size=*/15, /*alignment=*/1);
  EXPECT_NE(result, nullptr);
  EXPECT_EQ(allocator.GetUsedBytes(), static_cast<size_t>(20));
  EXPECT_EQ(allocator.GetRequestedBytes(), static_cast<size_t>(20));
  EXPECT_EQ(allocator.GetAllocatedCount(), static_cast<size_t>(1));
}

TEST(RecordingSingleArenaBufferAllocatorTest,
     TestRecordsMisalignedHeadSizeAdjustments) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::RecordingSingleArenaBufferAllocator allocator(arena, arena_size);
  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, 12);
  EXPECT_NE(resizable_buf, nullptr);

  EXPECT_EQ(kTfLiteOk, allocator.ResizeBuffer(resizable_buf, /*size=*/10,
                                              /*alignment=*/12));
  // Validate used bytes in 8 byte range that can included alignment of 12:
  EXPECT_GE(allocator.GetUsedBytes(), static_cast<size_t>(10));
  EXPECT_LE(allocator.GetUsedBytes(), static_cast<size_t>(20));
  EXPECT_EQ(allocator.GetRequestedBytes(), static_cast<size_t>(10));
  // Head adjustments do not count as an allocation:
  EXPECT_EQ(allocator.GetAllocatedCount(), static_cast<size_t>(0));
}

TEST(RecordingSingleArenaBufferAllocatorTest,
     TestDoesNotRecordFailedTailAllocations2) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::RecordingSingleArenaBufferAllocator allocator(arena, arena_size);

  uint8_t* resizable_buf = allocator.AllocateResizableBuffer(0, 1);
  EXPECT_NE(resizable_buf, nullptr);

  EXPECT_EQ(kTfLiteError,
            allocator.ResizeBuffer(resizable_buf,
                                   /*size=*/2048, /*alignment=*/1));
  EXPECT_EQ(allocator.GetUsedBytes(), static_cast<size_t>(0));
  EXPECT_EQ(allocator.GetRequestedBytes(), static_cast<size_t>(0));
  EXPECT_EQ(allocator.GetAllocatedCount(), static_cast<size_t>(0));
}

TF_LITE_MICRO_TESTS_MAIN
