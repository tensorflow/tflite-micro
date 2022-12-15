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

#include "tensorflow/lite/micro/micro_allocation_info.h"

#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestSingleSubgraph) {
  constexpr int kArenaSize = 1024;
  uint8_t arena[kArenaSize];
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  tflite::SingleArenaBufferAllocator allocator(arena, kArenaSize);
  tflite::AllocationInfoBuilder builder(model, &allocator);
  builder.CreateAllocationInfo(0);
  tflite::MicroAllocator* micro_allocator =
      tflite::MicroAllocator::Create(arena, kArenaSize);
  tflite::SubgraphAllocations* subgraph_allocations =
      micro_allocator->StartModelAllocation(model);
  builder.InitializeAllocationInfo(nullptr, subgraph_allocations);
  builder.MarkAllocationLifetimes(0, nullptr, nullptr, subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(builder.AllocationCount(), 4);
  tflite::AllocationInfo* allocation_info = builder.Finish();
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].last_used, 2);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].first_created, -1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].last_used, 2);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].first_created, 1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].last_used, 2);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].first_created, 2);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].last_used, 2);
}

TF_LITE_MICRO_TEST(TestSingleSubgraphWithIntermediates) {
  constexpr int kArenaSize = 1024;
  uint8_t arena[kArenaSize];
  const tflite::Model* model = tflite::testing::GetSimpleStatefulModel();
  tflite::SingleArenaBufferAllocator allocator(arena, kArenaSize);
  tflite::AllocationInfoBuilder builder(model, &allocator);
  builder.CreateAllocationInfo(0);
  tflite::MicroAllocator* micro_allocator =
      tflite::MicroAllocator::Create(arena, kArenaSize);
  tflite::SubgraphAllocations* subgraph_allocations =
      micro_allocator->StartModelAllocation(model);
  builder.InitializeAllocationInfo(nullptr, subgraph_allocations);
  builder.MarkAllocationLifetimes(0, nullptr, nullptr, subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(builder.AllocationCount(), 4);
  tflite::AllocationInfo* allocation_info = builder.Finish();
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].last_used, 1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].needs_allocating, true);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].first_created, 1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].last_used, 1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].needs_allocating, true);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].first_created, 1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].last_used, 1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].needs_allocating, true);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].first_created, -1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].last_used, -1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].needs_allocating, false);
}

TF_LITE_MICRO_TEST(TestMultiSubgraphWithIf) {
  constexpr int kArenaSize = 1024;
  uint8_t arena[kArenaSize];
  const tflite::Model* model =
      tflite::testing::GetSimpleModelWithSubgraphsAndIf();
  tflite::SingleArenaBufferAllocator allocator(arena, kArenaSize);
  tflite::AllocationInfoBuilder builder(model, &allocator);
  builder.CreateAllocationInfo(0);
  tflite::MicroAllocator* micro_allocator =
      tflite::MicroAllocator::Create(arena, kArenaSize);
  tflite::SubgraphAllocations* subgraph_allocations =
      micro_allocator->StartModelAllocation(model);
  builder.InitializeAllocationInfo(nullptr, subgraph_allocations);
  builder.MarkAllocationLifetimes(0, nullptr, nullptr, subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(builder.AllocationCount(), 10);
  tflite::AllocationInfo* allocation_info = builder.Finish();
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].last_used, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].last_used, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].last_used, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].first_created, 1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].last_used, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[4].first_created, 2);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[4].last_used, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[5].first_created, 2);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[5].last_used, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[6].first_created, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[6].last_used, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[7].first_created, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[7].last_used, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[8].first_created, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[8].last_used, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[9].first_created, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[9].last_used, 5);
}

TF_LITE_MICRO_TEST(TestMultiSubgraphWithIfAndEmptySubgraph) {
  constexpr int kArenaSize = 1024;
  uint8_t arena[kArenaSize];
  const tflite::Model* model =
      tflite::testing::GetSimpleModelWithIfAndEmptySubgraph();
  tflite::SingleArenaBufferAllocator allocator(arena, kArenaSize);
  tflite::AllocationInfoBuilder builder(model, &allocator);
  builder.CreateAllocationInfo(0);
  tflite::MicroAllocator* micro_allocator =
      tflite::MicroAllocator::Create(arena, kArenaSize);
  tflite::SubgraphAllocations* subgraph_allocations =
      micro_allocator->StartModelAllocation(model);
  builder.InitializeAllocationInfo(nullptr, subgraph_allocations);
  builder.MarkAllocationLifetimes(0, nullptr, nullptr, subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(builder.AllocationCount(), 10);
  tflite::AllocationInfo* allocation_info = builder.Finish();
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].last_used, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].last_used, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].last_used, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].first_created, 1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].last_used, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[4].first_created, 2);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[4].last_used, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[5].first_created, 2);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[5].last_used, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[6].first_created, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[6].last_used, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[7].first_created, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[7].last_used, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[8].first_created, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[8].last_used, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[9].first_created, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[9].last_used, 4);
}

TF_LITE_MICRO_TEST(TestMultiSubgraphWithIfAndInputSubgraphOverlap) {
  constexpr int kArenaSize = 2048;
  uint8_t arena[kArenaSize];
  const tflite::Model* model =
      tflite::testing::GetModelWithIfAndSubgraphInputTensorOverlap();
  tflite::SingleArenaBufferAllocator allocator(arena, kArenaSize);
  tflite::AllocationInfoBuilder builder(model, &allocator);
  builder.CreateAllocationInfo(0);
  tflite::MicroAllocator* micro_allocator =
      tflite::MicroAllocator::Create(arena, kArenaSize);
  tflite::SubgraphAllocations* subgraph_allocations =
      micro_allocator->StartModelAllocation(model);
  builder.InitializeAllocationInfo(nullptr, subgraph_allocations);
  builder.MarkAllocationLifetimes(0, nullptr, nullptr, subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(builder.AllocationCount(), 11);
  tflite::AllocationInfo* allocation_info = builder.Finish();
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[0].last_used, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[1].last_used, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].first_created, 0);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[2].last_used, 6);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].first_created, 1);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[3].last_used, 6);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[4].first_created, 6);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[4].last_used, 6);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[5].first_created, 2);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[5].last_used, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[6].first_created, 2);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[6].last_used, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[7].first_created, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[7].last_used, 3);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[8].first_created, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[8].last_used, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[9].first_created, 4);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[9].last_used, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[10].first_created, 5);
  TF_LITE_MICRO_EXPECT_EQ(allocation_info[10].last_used, 5);
}

TF_LITE_MICRO_TESTS_END
