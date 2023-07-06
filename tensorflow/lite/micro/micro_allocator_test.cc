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

#include "tensorflow/lite/micro/micro_allocator.h"

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_conv_model.h"

namespace tflite {
namespace testing {
namespace {

constexpr int t0 = 0;
constexpr int t1 = 1;
constexpr int t2 = 2;
constexpr int t3 = 3;
constexpr int t4 = 4;
constexpr int t5 = 5;

void VerifyMockTfLiteTensor(TfLiteTensor* tensor, bool is_variable = false) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(is_variable, tensor->is_variable);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), tensor->bytes);
  TF_LITE_MICRO_EXPECT(nullptr != tensor->data.raw);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          (reinterpret_cast<std::uintptr_t>(tensor->data.raw) %
                           MicroArenaBufferAlignment()));
}

// TODO(b/203663932): remove the usage of uint8 weight, which is deprecated.
void VerifyMockWeightTfLiteTensor(TfLiteTensor* tensor) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(1), tensor->bytes);
  TF_LITE_MICRO_EXPECT(nullptr != tensor->data.raw);
}

void VerifyMockTfLiteEvalTensor(TfLiteEvalTensor* tensor) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  size_t buffer_size;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::TfLiteEvalTensorByteLength(tensor, &buffer_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), buffer_size);
  TF_LITE_MICRO_EXPECT(nullptr != tensor->data.raw);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(0),
                          (reinterpret_cast<std::uintptr_t>(tensor->data.raw) %
                           MicroArenaBufferAlignment()));
}

void VerifyMockWeightTfLiteEvalTensor(TfLiteEvalTensor* tensor) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, tensor->type);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, tensor->dims->data[0]);
  size_t buffer_size;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::TfLiteEvalTensorByteLength(tensor, &buffer_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(1), buffer_size);
  TF_LITE_MICRO_EXPECT(nullptr != tensor->data.raw);
}

// TODO(b/203664378): rename to reflect the function does more than just verify.
void AllocateAndVerifyMockTensor(const Model* model, MicroAllocator* allocator,
                                 SubgraphAllocations* subgraph_allocations,
                                 int tensor_idx, bool is_variable = false) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
       subgraph_idx++) {
    VerifyMockTfLiteTensor(
        allocator->AllocatePersistentTfLiteTensor(model, subgraph_allocations,
                                                  tensor_idx, subgraph_idx),
        is_variable);
    VerifyMockTfLiteEvalTensor(
        &subgraph_allocations[subgraph_idx].tensors[tensor_idx]);
  }
}

void AllocateAndVerifyMockWeightTensor(
    const Model* model, MicroAllocator* allocator,
    SubgraphAllocations* subgraph_allocations, int tensor_idx) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
       subgraph_idx++) {
    VerifyMockWeightTfLiteTensor(allocator->AllocatePersistentTfLiteTensor(
        model, subgraph_allocations, tensor_idx, subgraph_idx));
    VerifyMockWeightTfLiteEvalTensor(
        &subgraph_allocations[subgraph_idx].tensors[tensor_idx]);
  }
}

void EnsureUniqueVariableTensorBuffer(const Model* model,
                                      TfLiteEvalTensor* eval_tensors,
                                      const int variable_tensor_idx) {
  for (size_t i = 0; i < GetModelTensorCount(model); i++) {
    if (i != static_cast<size_t>(variable_tensor_idx)) {
      TF_LITE_MICRO_EXPECT_NE(eval_tensors[variable_tensor_idx].data.raw,
                              eval_tensors[i].data.raw);
    }
  }
}

void VerifyRegistrationAndNodeAllocation(
    SubgraphAllocations* subgraph_allocations, size_t count,
    int num_subgraphs) {
  for (int subgraph_idx = 0; subgraph_idx < num_subgraphs; subgraph_idx++) {
    for (size_t i = 0; i < count; i++) {
      TF_LITE_MICRO_EXPECT(&subgraph_allocations[subgraph_idx]
                                .node_and_registrations[i]
                                .registration);
    }
  }
}

size_t GetArenaUsedBytesBySimpleMockModel(bool is_memory_planner_injected) {
  const int tensor_count = 4;
  const int node_count = 2;
  size_t eval_tensor_size = AlignSizeUp<TfLiteEvalTensor>(tensor_count);
  size_t node_registration_size = AlignSizeUp<NodeAndRegistration>(node_count);

  const int activation_tensor_count = 3;
  size_t activation_tensor_buffer =
      activation_tensor_count * AlignSizeUp(1, MicroArenaBufferAlignment());

  size_t default_tail_usage =
      MicroAllocator::GetDefaultTailUsage(/*is_memory_plan_given=*/false);
  if (is_memory_planner_injected) {
    default_tail_usage =
        MicroAllocator::GetDefaultTailUsage(/*is_memory_plan_given=*/true);
  }

  return default_tail_usage + eval_tensor_size + node_registration_size +
         activation_tensor_buffer;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInitializeRuntimeTensor) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SingleArenaBufferAllocator* simple_allocator =
      tflite::SingleArenaBufferAllocator::Create(arena, arena_size);

  const tflite::Tensor* tensor = tflite::testing::Create1dFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
          simple_allocator, simple_allocator, /*allocate_temp=*/false, *tensor,
          buffers, &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(400), allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT(nullptr == allocated_tensor.data.i32);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteArenaRw, allocated_tensor.allocation_type);

  simple_allocator->~SingleArenaBufferAllocator();
}

// TODO(b/162311891): Drop this test when InitializeTfLiteTensorFromFlatbuffer()
// always allocates from temp (interpreter returns buffers from
// TfLiteEvalTensor):
TF_LITE_MICRO_TEST(TestInitializeTempRuntimeTensor) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SingleArenaBufferAllocator* simple_allocator =
      tflite::SingleArenaBufferAllocator::Create(arena, arena_size);

  const tflite::Tensor* tensor = tflite::testing::Create1dFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_temp_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
                     simple_allocator, simple_allocator, /*allocate_temp=*/true,
                     *tensor, buffers, &allocated_temp_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_temp_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_temp_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_temp_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(400),
                          allocated_temp_tensor.bytes);
  TF_LITE_MICRO_EXPECT(nullptr == allocated_temp_tensor.data.i32);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteArenaRw,
                          allocated_temp_tensor.allocation_type);

  simple_allocator->~SingleArenaBufferAllocator();
}

TF_LITE_MICRO_TEST(TestInitializeQuantizedTensor) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SingleArenaBufferAllocator* simple_allocator =
      tflite::SingleArenaBufferAllocator::Create(arena, arena_size);

  const tflite::Tensor* tensor =
      tflite::testing::CreateQuantizedFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
          simple_allocator, simple_allocator, /*allocate_temp=*/false, *tensor,
          buffers, &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(400), allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT(nullptr == allocated_tensor.data.i32);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteArenaRw, allocated_tensor.allocation_type);

  simple_allocator->~SingleArenaBufferAllocator();
}

TF_LITE_MICRO_TEST(TestMissingQuantization) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::SingleArenaBufferAllocator* simple_allocator =
      tflite::SingleArenaBufferAllocator::Create(arena, arena_size);

  const tflite::Tensor* tensor =
      tflite::testing::CreateMissingQuantizationFlatbufferTensor(100);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      tflite::testing::CreateFlatbufferBuffers();

  TfLiteTensor allocated_tensor;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::internal::InitializeTfLiteTensorFromFlatbuffer(
          simple_allocator, simple_allocator, /*allocate_temp=*/false, *tensor,
          buffers, &allocated_tensor));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, allocated_tensor.type);
  TF_LITE_MICRO_EXPECT_EQ(1, allocated_tensor.dims->size);
  TF_LITE_MICRO_EXPECT_EQ(100, allocated_tensor.dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(400), allocated_tensor.bytes);
  TF_LITE_MICRO_EXPECT(nullptr == allocated_tensor.data.i32);
}

TF_LITE_MICRO_TEST(TestFailsWhenModelStartsTwice) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  TF_LITE_MICRO_EXPECT(nullptr != allocator->StartModelAllocation(model));
  TF_LITE_MICRO_EXPECT(nullptr == allocator->StartModelAllocation(model));
}

TF_LITE_MICRO_TEST(TestFailsWithWrongSequence) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  tflite::SubgraphAllocations* subgraph_allocations = nullptr;
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);

  // We can't finish allocation before it ever got started.
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, allocator->FinishModelAllocation(
                        model, subgraph_allocations, &scratch_buffer_handles));

  // Start twice is not allowed.
  TF_LITE_MICRO_EXPECT(nullptr != allocator->StartModelAllocation(model));
  TF_LITE_MICRO_EXPECT(nullptr == allocator->StartModelAllocation(model));
}

TF_LITE_MICRO_TEST(TestMockModelAllocation) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  constexpr size_t arena_size = 1024 + 16;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));
  size_t expected_arena_used_bytes =
      tflite::testing::GetArenaUsedBytesBySimpleMockModel(
          /*is_memory_planner_injected=*/false);
  TF_LITE_MICRO_EXPECT_EQ(allocator->used_bytes(), expected_arena_used_bytes);

  size_t model_tensor_size = tflite::testing::GetModelTensorCount(model);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), model_tensor_size);

  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 0);
  tflite::testing::AllocateAndVerifyMockWeightTensor(model, allocator,
                                                     subgraph_allocations, 1);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 2);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 3);

  TfLiteEvalTensor* eval_tensors = subgraph_allocations[0].tensors;
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[1].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[2].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[1].data.raw, eval_tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[1].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[2].data.raw);

  // SimpleMockModel has 2 operators:
  tflite::testing::VerifyRegistrationAndNodeAllocation(subgraph_allocations,
                                                       /*count=*/2,
                                                       /*num_subgraphs=*/1);
}

TF_LITE_MICRO_TEST(TestMockModelAllocationInTwoSeparateArenas) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  constexpr size_t arena_size = 1024;
  uint8_t persistent_arena[arena_size];
  uint8_t non_persistent_arena[arena_size];

  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      persistent_arena, arena_size, non_persistent_arena, arena_size);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  size_t model_tensor_size = tflite::testing::GetModelTensorCount(model);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), model_tensor_size);

  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 0);
  tflite::testing::AllocateAndVerifyMockWeightTensor(model, allocator,
                                                     subgraph_allocations, 1);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 2);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 3);

  TfLiteEvalTensor* eval_tensors = subgraph_allocations[0].tensors;
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[1].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[2].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[1].data.raw, eval_tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[1].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[2].data.raw);

  // SimpleMockModel has 2 operators:
  tflite::testing::VerifyRegistrationAndNodeAllocation(subgraph_allocations,
                                                       /*count=*/2,
                                                       /*num_subgraphs=*/1);
}

TF_LITE_MICRO_TEST(TestMockModelAllocationWithGivenMemoryPlanner) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::GreedyMemoryPlanner memory_planner;
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size, &memory_planner);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));
  size_t expected_arena_used_bytes =
      tflite::testing::GetArenaUsedBytesBySimpleMockModel(
          /*is_memory_planner_injected=*/true);
  TF_LITE_MICRO_EXPECT_EQ(allocator->used_bytes(), expected_arena_used_bytes);

  size_t model_tensor_size = tflite::testing::GetModelTensorCount(model);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), model_tensor_size);

  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 0);
  tflite::testing::AllocateAndVerifyMockWeightTensor(model, allocator,
                                                     subgraph_allocations, 1);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 2);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 3);

  TfLiteEvalTensor* eval_tensors = subgraph_allocations[0].tensors;
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[1].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[2].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[1].data.raw, eval_tensors[2].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[0].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[1].data.raw);
  TF_LITE_MICRO_EXPECT_NE(eval_tensors[3].data.raw, eval_tensors[2].data.raw);

  // SimpleMockModel has 2 operators:
  tflite::testing::VerifyRegistrationAndNodeAllocation(subgraph_allocations,
                                                       /*count=*/2,
                                                       /*num_subgraphs=*/1);
}

TF_LITE_MICRO_TEST(TestMultiTenantAllocation) {
  // The `OpResolver` is shared among different models in this test for
  // simplicity but in practice you could have different `OpResolver`.
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));

  // Create a shared allocator.
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);

  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;

  // Allocate for model 1. We use ComplexMockModel here to cover the code path
  // allocatig variables.
  const tflite::Model* model1 = tflite::testing::GetComplexMockModel();
  tflite::SubgraphAllocations* subgraph_allocations1 =
      allocator->StartModelAllocation(model1);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations1);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model1, subgraph_allocations1,
                                                  &scratch_buffer_handles));
  const size_t single_model_used_bytes = allocator->used_bytes();

  // Allocate for model 2.
  const tflite::Model* model2 = tflite::testing::GetComplexMockModel();
  tflite::SubgraphAllocations* subgraph_allocations2 =
      allocator->StartModelAllocation(model2);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations2);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model2, subgraph_allocations2,
                                                  &scratch_buffer_handles));

  // Allocation for two instances of the same model takes less memory as `head`
  // of the arena is reused.
  TF_LITE_MICRO_EXPECT_LE(allocator->used_bytes(), 2 * single_model_used_bytes);
}

TF_LITE_MICRO_TEST(TestMultiTenantAllocationInTwoSeparateArenas) {
  // The `OpResolver` is shared among different models in this test for
  // simplicity but in practice you could have different `OpResolver`.
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));

  // Create a shared allocator.
  constexpr size_t arena_size = 4096;
  uint8_t persistent_arena[arena_size];
  uint8_t non_persistent_arena[arena_size];

  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(
      persistent_arena, arena_size, non_persistent_arena, arena_size);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;

  // Allocate for model 1. We use ComplexMockModel here to cover the code path
  // allocatig variables.
  const tflite::Model* model1 = tflite::testing::GetComplexMockModel();
  tflite::SubgraphAllocations* subgraph_allocations1 =
      allocator->StartModelAllocation(model1);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations1);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model1, subgraph_allocations1,
                                                  &scratch_buffer_handles));
  const size_t single_model_used_bytes = allocator->used_bytes();

  // Allocate for model 2.
  const tflite::Model* model2 = tflite::testing::GetComplexMockModel();
  tflite::SubgraphAllocations* subgraph_allocations2 =
      allocator->StartModelAllocation(model2);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations2);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model2, subgraph_allocations2,
                                                  &scratch_buffer_handles));

  // Allocation for two instances of the same model takes less memory as `head`
  // of the arena is reused.
  TF_LITE_MICRO_EXPECT_LE(allocator->used_bytes(), 2 * single_model_used_bytes);
}

TF_LITE_MICRO_TEST(TestAllocationForModelsWithBranches) {
  const tflite::Model* model = tflite::testing::GetSimpleModelWithBranch();
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  int8_t* start =
      tflite::micro::GetTensorData<int8_t>(&subgraph_allocations[0].tensors[0]);
  // Check test_helpers.cc BuildSimpleModelWithBranch for model structure.
  // t0 is the first tensor, so place it in offset 0.
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[0]) -
                                 start);
  // bytes = 2 * 2 * 3 * sizeof(float32) = 48, same for other tensors.
  size_t buffer_size;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::TfLiteEvalTensorByteLength(
                     &subgraph_allocations[0].tensors[0], &buffer_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(48), buffer_size);
  // t1 can't reuse any memory, as n0 requires both t0 and t1.
  TF_LITE_MICRO_EXPECT_EQ(96, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[1]) -
                                  start);
  // t2 can't reuse any memory, as n1 requires both t0 and t2. Also n2 requires
  // both t1 and t2.
  TF_LITE_MICRO_EXPECT_EQ(48, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[2]) -
                                  start);
  // t3 reuses the same memory from t0 as t0 is not an input to any node.
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[3]) -
                                 start);

  // SimpleModelWithBranch has 3 operators:
  tflite::testing::VerifyRegistrationAndNodeAllocation(subgraph_allocations,
                                                       /*count=*/3,
                                                       /*num_subgraphs=*/1);
}

TF_LITE_MICRO_TEST(TestAllocationForComplexModelAllocation) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  constexpr size_t arena_size = 2048;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);
  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  size_t model_tensor_size = tflite::testing::GetModelTensorCount(model);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(10), model_tensor_size);

  // NOTE: Tensor indexes match the values in GetComplexMockModel().
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 0);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 1,
                                               /*is_variable=*/
                                               true);
  tflite::testing::AllocateAndVerifyMockWeightTensor(model, allocator,
                                                     subgraph_allocations, 2);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 3);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 4,
                                               /*is_variable=*/
                                               true);
  tflite::testing::AllocateAndVerifyMockWeightTensor(model, allocator,
                                                     subgraph_allocations, 5);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 6);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 7,
                                               /*is_variable=*/
                                               true);
  tflite::testing::AllocateAndVerifyMockWeightTensor(model, allocator,
                                                     subgraph_allocations, 8);
  tflite::testing::AllocateAndVerifyMockTensor(model, allocator,
                                               subgraph_allocations, 9);

  // // Ensure that variable tensors have unique address
  tflite::testing::EnsureUniqueVariableTensorBuffer(
      model, subgraph_allocations[0].tensors, 1);
  tflite::testing::EnsureUniqueVariableTensorBuffer(
      model, subgraph_allocations[0].tensors, 4);
  tflite::testing::EnsureUniqueVariableTensorBuffer(
      model, subgraph_allocations[0].tensors, 7);

  // ComplexMockModel has 3 operators:
  tflite::testing::VerifyRegistrationAndNodeAllocation(subgraph_allocations,
                                                       /*count=*/3,
                                                       /*num_subgraphs=*/1);
}

TF_LITE_MICRO_TEST(OfflinePlannerBranchesAllOnline) {
  int version = 1;
  int subgraph = 0;
  constexpr int number_tensors = 4;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {version, subgraph,
                                                   number_tensors,  // header
                                                   // memory offsets:
                                                   -1, -1, -1, -1};

  // The structure is identical to the one in
  // TestAllocationForModelsWithBranches
  int number_connections = 3;
  tflite::testing::NodeConnection node_list[3] = {{
                                                      {0},  // input
                                                      {1}   // output
                                                  },
                                                  {
                                                      {0},  // input
                                                      {2}   // output
                                                  },
                                                  {
                                                      {1, 2},  // input1, input2
                                                      {3}      // output
                                                  }};

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections);

  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;

  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);

  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  // Since all of the tensors are online planned and the model structure is
  // identical to that in TestAllocationForModelsWithBranches,
  // the offsets be should identical to that test.
  int8_t* start =
      tflite::micro::GetTensorData<int8_t>(&subgraph_allocations[0].tensors[0]);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[0]) -
                                 start);

  size_t buffer_size;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::TfLiteEvalTensorByteLength(
                     &subgraph_allocations[0].tensors[0], &buffer_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(48), buffer_size);
  TF_LITE_MICRO_EXPECT_EQ(96, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[1]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(48, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[2]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[3]) -
                                 start);
}

TF_LITE_MICRO_TEST(OfflinePlannerBasic) {
  constexpr int number_tensors = 4;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {1,         0, number_tensors,
                                                   /*t0=*/0,
                                                   /*t1=*/48,
                                                   /*t2=*/0,
                                                   /*t3=*/48};
  constexpr int number_connections = 3;
  tflite::testing::NodeConnection node_list[number_connections] = {
      {/*input=*/{tflite::testing::t0},
       /*output=*/{tflite::testing::t1}},
      {/*input=*/{tflite::testing::t1},
       /*output=*/{tflite::testing::t2}},
      {/*input=*/{tflite::testing::t2},
       /*output=*/{tflite::testing::t3}}};

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections);

  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);

  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  int8_t* start =
      tflite::micro::GetTensorData<int8_t>(&subgraph_allocations[0].tensors[0]);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[0]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(48, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[1]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[2]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(48, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[3]) -
                                  start);
}

TF_LITE_MICRO_TEST(OfflinePlannerOverlappingAllocation) {
  constexpr int number_tensors = 4;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {/*version=*/1,
                                                   /*subgraph=*/0,
                                                   number_tensors,
                                                   /*t0=*/0,
                                                   /*t1=*/0,
                                                   /*t2=*/48,
                                                   /*t3=*/-1};

  int number_connections = 2;
  tflite::testing::NodeConnection node_list[2] = {
      {/*input, scratch=*/{tflite::testing::t0, tflite::testing::t1},
       /*output=*/{tflite::testing::t2}},
      {/*input=*/{tflite::testing::t2},
       /*output=*/{tflite::testing::t3}},
  };

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections);

  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);

  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  int8_t* start =
      tflite::micro::GetTensorData<int8_t>(&subgraph_allocations[0].tensors[0]);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[0]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[1]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(48, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[2]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[3]) -
                                 start);
  // TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(48), context.tensors[0].bytes);
}

TF_LITE_MICRO_TEST(OfflinePlannerOfflineOnline) {
  constexpr int number_tensors = 5;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {/*version=*/1,
                                                   /*subgraph=*/0,
                                                   number_tensors,
                                                   /*t0=*/0,
                                                   /*t1=*/48,
                                                   /*t2=*/-1,
                                                   /*t3=*/0,
                                                   /*t4=*/-1};

  constexpr int number_connections = 2;
  tflite::testing::NodeConnection node_list[number_connections] = {
      {
          /*input, scratch=*/{tflite::testing::t0, tflite::testing::t1},
          /*output=*/{tflite::testing::t2},
      },
      {
          /*input=*/{tflite::testing::t2},
          /*output1, output2=*/{tflite::testing::t3, tflite::testing::t4},
      },
  };

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections);

  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);

  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  int8_t* start =
      tflite::micro::GetTensorData<int8_t>(&subgraph_allocations[0].tensors[0]);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[0]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(48, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[1]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(96, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[2]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(48, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[4]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[3]) -
                                 start);
}

TF_LITE_MICRO_TEST(TestAllocatePersistentTfLiteTensor) {
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  constexpr size_t arena_size = 1024 * 12;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(allocator != nullptr);

  TfLiteTensor* tensor1 = allocator->AllocatePersistentTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, /*tensor_index=*/1,
      /*subgraph_index=*/0);
  TF_LITE_MICRO_EXPECT(tensor1 != nullptr);
  TF_LITE_MICRO_EXPECT(tensor1->quantization.params != nullptr);
  TF_LITE_MICRO_EXPECT_FALSE(tensor1->is_variable);

  TfLiteTensor* tensor2 = allocator->AllocatePersistentTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, /*tensor_index=*/2,
      /*subgraph_index=*/0);
  TF_LITE_MICRO_EXPECT(tensor2 != nullptr);
  TF_LITE_MICRO_EXPECT(tensor2->quantization.params != nullptr);
  TF_LITE_MICRO_EXPECT_FALSE(tensor2->is_variable);

  // The address of tensor1 should be higher than the address of tensor2 since
  // persistent allocations take place in the tail which grows downward.
  TF_LITE_MICRO_EXPECT_GT(tensor1, tensor2);
}

TF_LITE_MICRO_TEST(TestFailAllocatePersistentTfLiteTensor) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  // MicroAllocator::Create always allocates GreedyMemoryPlanner,
  // SingleArenaBufferAllocator and MicroAllocator objects.
  // Memory available should be <= the sum of the alignments which
  // is < sizeof(TfLiteTensor).
  constexpr size_t kArenaSize = sizeof(tflite::GreedyMemoryPlanner) +
                                alignof(tflite::GreedyMemoryPlanner) +
                                sizeof(tflite::MicroAllocator) +
                                alignof(tflite::MicroAllocator) +
                                sizeof(tflite::SingleArenaBufferAllocator) +
                                alignof(tflite::SingleArenaBufferAllocator) +
                                tflite::MicroArenaBufferAlignment();
  uint8_t arena[kArenaSize];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, sizeof(arena));
  TF_LITE_MICRO_EXPECT(allocator != nullptr);

  TfLiteTensor* tensor1 = allocator->AllocatePersistentTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, /*tensor_index=*/1,
      /*subgraph_index=*/0);
  TF_LITE_MICRO_EXPECT(tensor1 == nullptr);
}

TF_LITE_MICRO_TEST(TestAllocateSingleTempTfLiteTensor) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(allocator != nullptr);

  TfLiteTensor* tensor1 = allocator->AllocateTempTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, /*tensor_index=*/1,
      /*subgraph_index=*/0);
  TF_LITE_MICRO_EXPECT(tensor1 != nullptr);
}

TF_LITE_MICRO_TEST(TestAllocateChainOfTfLiteTensor) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(allocator != nullptr);

  TfLiteTensor* tensor1 = allocator->AllocateTempTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, /*tensor_index=*/1,
      /*subgraph_index=*/0);
  TF_LITE_MICRO_EXPECT(tensor1 != nullptr);

  TfLiteTensor* tensor2 = allocator->AllocateTempTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, /*tensor_index=*/2,
      /*subgraph_index=*/0);
  TF_LITE_MICRO_EXPECT(tensor2 != nullptr);

  // The address of tensor2 should be higher than the address of tensor1
  // (chained allocations):
  TF_LITE_MICRO_EXPECT_GT(tensor2, tensor1);
}

TF_LITE_MICRO_TEST(TestAllocateAndDeallocateChainOfTfLiteTensor) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(allocator != nullptr);

  TfLiteTensor* tensor1 = allocator->AllocateTempTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, /*tensor_index=*/1,
      /*subgraph_index=*/0);
  TF_LITE_MICRO_EXPECT(tensor1 != nullptr);

  TfLiteTensor* tensor2 = allocator->AllocateTempTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, /*tensor_index=*/2,
      /*subgraph_index=*/0);
  TF_LITE_MICRO_EXPECT(tensor2 != nullptr);

  // The address of tensor2 should be higher than the address of tensor1
  // (chained allocations):
  TF_LITE_MICRO_EXPECT_GT(tensor2, tensor1);

  // Deallocate only one temp TfLiteTensor does not deallocate all temp buffers.
  allocator->DeallocateTempTfLiteTensor(tensor1);
  TF_LITE_MICRO_EXPECT_FALSE(allocator->IsAllTempDeallocated());

  // Deallocate both temp TfLiteTensor deallocate all temp buffers.
  allocator->DeallocateTempTfLiteTensor(tensor2);
  TF_LITE_MICRO_EXPECT_TRUE(allocator->IsAllTempDeallocated());
}

TF_LITE_MICRO_TEST(TestAllocateAndDeallocateTempBuffer) {
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(allocator != nullptr);
  TF_LITE_MICRO_EXPECT_TRUE(allocator->IsAllTempDeallocated());
  uint8_t* buffer1 =
      allocator->AllocateTempBuffer(10, tflite::MicroArenaBufferAlignment());
  TF_LITE_MICRO_EXPECT(buffer1 != nullptr);
  TF_LITE_MICRO_EXPECT_FALSE(allocator->IsAllTempDeallocated());
  allocator->DeallocateTempBuffer(buffer1);
  TF_LITE_MICRO_EXPECT_TRUE(allocator->IsAllTempDeallocated());
}

TF_LITE_MICRO_TEST(TestAllocateTfLiteTensorWithReset) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(allocator != nullptr);

  TfLiteTensor* tensor1 = allocator->AllocateTempTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, /*tensor_index=*/1,
      /*subgraph_index=*/0);
  TF_LITE_MICRO_EXPECT(tensor1 != nullptr);

  allocator->DeallocateTempTfLiteTensor(tensor1);

  allocator->ResetTempAllocations();

  TfLiteTensor* tensor2 = allocator->AllocateTempTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, /*tensor_index=*/2,
      /*subgraph_index=*/0);
  TF_LITE_MICRO_EXPECT(tensor2 != nullptr);

  // The address of tensor2 should be equal than the address of tensor1 since
  // allocations were not chained:
  TF_LITE_MICRO_EXPECT(tensor2 == tensor1);
}

TF_LITE_MICRO_TEST(TestOperatorInputsNotInSubgraphInputs) {
  constexpr int number_tensors = 5;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {/*version=*/1,
                                                   /*subgraph=*/0,
                                                   number_tensors,
                                                   /*t0=*/0,
                                                   /*t1=*/0,
                                                   /*t2=*/0,
                                                   /*t3=*/48,
                                                   /*t4=*/-1};

  constexpr int number_connections = 2;
  tflite::testing::NodeConnection node_list[number_connections] = {
      {// t0: input (actual input part of subgraph inputs as
       // well as operator inputs)
       // t1: scratch1 (only in operator inputs)
       // t2: scratch2 (only in operator inputs)
       {tflite::testing::t0, tflite::testing::t1, tflite::testing::t2},
       /*t3: output=*/{tflite::testing::t3}},
      {/*t3: input=*/{tflite::testing::t3},
       /*t4: output=*/{tflite::testing::t4}},
  };

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections,
      /*Only first tensor (t0) is in subgraph input list=*/1);

  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);

  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  int8_t* start =
      tflite::micro::GetTensorData<int8_t>(&subgraph_allocations[0].tensors[0]);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[0]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[1]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[2]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(48, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[3]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[4]) -
                                 start);
}

TF_LITE_MICRO_TEST(TestTypicalFirstOpAndSecondOpWithScratchTensors) {
  constexpr int number_tensors = 6;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  const int32_t metadata_buffer[tflite::testing::kOfflinePlannerHeaderSize +
                                number_tensors] = {/*version=*/1,
                                                   /*subgraph=*/0,
                                                   number_tensors,
                                                   /*t0=*/0,
                                                   /*t1=*/0,
                                                   /*t2=*/0,
                                                   /*t3=*/0,
                                                   /*t4=*/48,
                                                   /*t5=*/-1};

  constexpr int number_connections = 3;
  tflite::testing::NodeConnection node_list[number_connections] = {
      {/*t0: input (subgraph and operator input)=*/{tflite::testing::t0},
       /*t1: output=*/{tflite::testing::t1}},
      {// t1: input
       // t2: scratch1 (only in operator inputs)
       // t3: scratch2 (only in operator inputs)
       {tflite::testing::t1, tflite::testing::t2, tflite::testing::t3},

       /*t4: output=*/{tflite::testing::t4}},
      {/*t4: input=*/{tflite::testing::t4},
       /*t5: output=*/{tflite::testing::t5}},
  };

  const tflite::Model* model = tflite::testing::GetModelWithOfflinePlanning(
      number_tensors, metadata_buffer, node_list, number_connections,
      /*Only first tensor (t0) is in subgraph input list=*/1);

  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);

  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  int8_t* start =
      tflite::micro::GetTensorData<int8_t>(&subgraph_allocations[0].tensors[0]);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[0]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[1]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[2]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[3]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(48, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[4]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[5]) -
                                 start);
}

TF_LITE_MICRO_TEST(TestModelWithUnusedTensors) {
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));

  const tflite::Model* model = tflite::testing::GetModelWithUnusedInputs();

  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);

  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  // Unused input tensor should not occupy any space.
  int8_t* start =
      tflite::micro::GetTensorData<int8_t>(&subgraph_allocations[0].tensors[2]);
  TF_LITE_MICRO_EXPECT_EQ(64, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[0]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[1]) -
                                 start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[2]) -
                                 start);
  // Unused tensor should not occupy any space.
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[3]) -
                                 start);
}

TF_LITE_MICRO_TEST(TestModelWithUnusedOperatorOutputs) {
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));

  const tflite::Model* model =
      tflite::testing::GetModelWithUnusedOperatorOutputs();

  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  constexpr size_t arena_size = 4096;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size);

  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  // Unused output tensor should have its own allocation.
  int8_t* start =
      tflite::micro::GetTensorData<int8_t>(&subgraph_allocations[0].tensors[1]);
  TF_LITE_MICRO_EXPECT_EQ(64, tflite::micro::GetTensorData<int8_t>(
                                  &subgraph_allocations[0].tensors[0]) -
                                  start);
  TF_LITE_MICRO_EXPECT_EQ(0, tflite::micro::GetTensorData<int8_t>(
                                 &subgraph_allocations[0].tensors[1]) -
                                 start);
}

// Manually create an offline plan for the SimpleMockModel. Pass that into the
// interpreter and confirm that the eval tensors' offsets are exactly what was
// specified in the offline plan.
TF_LITE_MICRO_TEST(TestMockModelAllocationByNonPersistentMemoryPlannerShim) {
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();

  // The simple model has three activation tensors 0, 2, 3, which corresponding
  // to buffer request 0, 1, 2.
  constexpr size_t kBufferEntriesCount = 3;
  constexpr size_t kBufferPlanSize =
      sizeof(tflite::BufferPlan) +
      (kBufferEntriesCount) * sizeof(tflite::BufferDescriptor);
  // The offsets of buffers are chosen to be very different from what the
  // default greedy memory planner would select to reflect that buffers does NOT
  // need to start at offset 0 and in contiguous order. The offsets in a given
  // memory plan just need to meet the 4-byte buffer alignment requirement from
  // the framework side. The memory plan provider guarantees the correctness
  // of the plan for the model.
  constexpr int32_t kOffset0 = 200;
  constexpr int32_t kOffset1 = 64;
  constexpr int32_t kOffset2 = 120;
  // Allocate a memory plan buffer first b/c the struct BufferPlan has a
  // flexible member array.
  uint8_t buffer_plan_arena[kBufferPlanSize];
  tflite::BufferPlan* non_persistent_buffer_plan =
      reinterpret_cast<tflite::BufferPlan*>(buffer_plan_arena);
  non_persistent_buffer_plan->buffer_count = kBufferEntriesCount;
  non_persistent_buffer_plan->buffer_plan_entries[0].offset = kOffset0;
  non_persistent_buffer_plan->buffer_plan_entries[1].offset = kOffset1;
  non_persistent_buffer_plan->buffer_plan_entries[2].offset = kOffset2;

  tflite::NonPersistentMemoryPlannerShim planner(non_persistent_buffer_plan);

  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite::testing::GetTestingOpResolver(op_resolver));
  constexpr size_t arena_size = 1024;
  uint8_t arena[arena_size];
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(arena, arena_size, &planner);
  TF_LITE_MICRO_EXPECT(allocator != nullptr);
  tflite::SubgraphAllocations* subgraph_allocations =
      allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  size_t model_tensor_size = tflite::testing::GetModelTensorCount(model);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), model_tensor_size);
  tflite::testing::AllocateAndVerifyMockWeightTensor(model, allocator,
                                                     subgraph_allocations, 1);

  TfLiteEvalTensor* eval_tensors = subgraph_allocations[0].tensors;

  // Offset is relative to the arena after the buffer alignment adjustment which
  // happens when MicroAllocator is created.
  uint8_t* aligned_arena =
      tflite::AlignPointerUp(arena, tflite::MicroArenaBufferAlignment());

  TF_LITE_MICRO_EXPECT_TRUE(static_cast<uint8_t*>(eval_tensors[0].data.data) ==
                            (aligned_arena + kOffset0));
  TF_LITE_MICRO_EXPECT_TRUE(static_cast<uint8_t*>(eval_tensors[2].data.data) ==
                            (aligned_arena + kOffset1));
  TF_LITE_MICRO_EXPECT_TRUE(static_cast<uint8_t*>(eval_tensors[3].data.data) ==
                            (aligned_arena + kOffset2));

  // SimpleMockModel has 2 operators:
  tflite::testing::VerifyRegistrationAndNodeAllocation(subgraph_allocations,
                                                       /*count=*/2,
                                                       /*num_subgraphs=*/1);
}

TF_LITE_MICRO_TEST(TestMultiSubgraphNumScratchAllocations) {
  // Any test model with multiple subgraphs will suffice
  const tflite::Model* model =
      tflite::testing::GetSimpleModelWithNullInputsAndOutputs();

  constexpr size_t arena_size = 2048 * 2;
  uint8_t arena[arena_size];

  tflite::MicroAllocator* allocator = nullptr;
  tflite::SubgraphAllocations* subgraph_allocations = nullptr;
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;

  // First iteration: no scratch buffers
  allocator = tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);

  subgraph_allocations = allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  size_t used_bytes = allocator->used_bytes();

  // Second iteration: the same but request two scratch buffers
  tflite::MicroAllocator::Create(arena, arena_size);
  TF_LITE_MICRO_EXPECT(nullptr != allocator);

  subgraph_allocations = allocator->StartModelAllocation(model);
  TF_LITE_MICRO_EXPECT(nullptr != subgraph_allocations);

  // Request two scratch buffers.
  // They have size 0 because we do not want to affect the memory plan
  const int scratch_subgraph_idx = 0;
  const int scratch_node_idx = 0;
  const size_t scratch_size = 0;
  int buffer_idx1 = -1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->RequestScratchBufferInArena(
                     scratch_size, scratch_subgraph_idx, &buffer_idx1));
  int buffer_idx2 = -1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->RequestScratchBufferInArena(
                     scratch_size, scratch_subgraph_idx, &buffer_idx2));
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishPrepareNodeAllocations(scratch_node_idx));

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles));

  // Check that AllocateScratchBufferHandles was only called once, i.e. only two
  // tflite::ScratchBufferHandle should have been allocated.
  size_t used_bytes_with_scratch = allocator->used_bytes();

  TF_LITE_MICRO_EXPECT_EQ(used_bytes_with_scratch,
                          used_bytes + sizeof(tflite::ScratchBufferHandle) * 2);
}

TF_LITE_MICRO_TESTS_END
