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

#include "tensorflow/lite/micro/recording_micro_allocator.h"

#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"
#include "tensorflow/lite/micro/testing/test_conv_model.h"

#define TF_LITE_TENSOR_STRUCT_SIZE sizeof(TfLiteTensor)
#define TF_LITE_EVAL_TENSOR_STRUCT_SIZE sizeof(TfLiteEvalTensor)
#define TF_LITE_AFFINE_QUANTIZATION_SIZE sizeof(TfLiteAffineQuantization)
#define NODE_AND_REGISTRATION_STRUCT_SIZE sizeof(tflite::NodeAndRegistration)

// TODO(b/158303868): Move tests into anonymous namespace.
namespace {

constexpr int kTestConvArenaSize = 1024 * 12;

}  // namespace

TEST(RecordingMicroAllocatorTest, TestRecordsTfLiteEvalTensorArrayData) {
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver ops_resolver;
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize);
  // TODO(b/158102673): ugly workaround for not having fatal assertions. Same
  // throughout this file.
  ASSERT_NE(micro_allocator, nullptr);

  tflite::SubgraphAllocations* subgraph_allocations =
      micro_allocator->StartModelAllocation(model);
  ASSERT_NE(subgraph_allocations, nullptr);

  TfLiteStatus status = micro_allocator->FinishModelAllocation(
      model, subgraph_allocations, &scratch_buffer_handles);
  ASSERT_EQ(status, kTfLiteOk);

  micro_allocator->PrintAllocations();

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kTfLiteEvalTensorData);

  micro_allocator->PrintAllocations();

  size_t tensors_count = tflite::testing::GetModelTensorCount(model);

  EXPECT_EQ(recorded_allocation.count, tensors_count);
  EXPECT_EQ(recorded_allocation.requested_bytes,
            tensors_count * TF_LITE_EVAL_TENSOR_STRUCT_SIZE);
  EXPECT_GE(recorded_allocation.used_bytes,
            tensors_count * TF_LITE_EVAL_TENSOR_STRUCT_SIZE);
}

TEST(RecordingMicroAllocatorTest, TestRecordsNodeAndRegistrationArrayData) {
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver ops_resolver;
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize);
  ASSERT_NE(micro_allocator, nullptr);

  tflite::SubgraphAllocations* subgraph_allocations =
      micro_allocator->StartModelAllocation(model);
  ASSERT_NE(nullptr, subgraph_allocations);

  TfLiteStatus status = micro_allocator->FinishModelAllocation(
      model, subgraph_allocations, &scratch_buffer_handles);
  ASSERT_EQ(status, kTfLiteOk);

  size_t num_ops = model->subgraphs()->Get(0)->operators()->size();
  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kNodeAndRegistrationArray);

  EXPECT_EQ(recorded_allocation.count, num_ops);
  EXPECT_EQ(recorded_allocation.requested_bytes,
            num_ops * NODE_AND_REGISTRATION_STRUCT_SIZE);
  EXPECT_GE(recorded_allocation.used_bytes,
            num_ops * NODE_AND_REGISTRATION_STRUCT_SIZE);
}

TEST(RecordingMicroAllocatorTest, TestRecordsMultiTenantAllocations) {
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver ops_resolver;
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);

  // Double the arena size to allocate two models inside of it:
  uint8_t arena[kTestConvArenaSize * 2];

  TfLiteStatus status;

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize * 2);
  ASSERT_NE(micro_allocator, nullptr);

  // First allocation with the model in the arena:
  tflite::SubgraphAllocations* subgraph_allocations =
      micro_allocator->StartModelAllocation(model);
  ASSERT_NE(nullptr, subgraph_allocations);

  status = micro_allocator->FinishModelAllocation(model, subgraph_allocations,
                                                  &scratch_buffer_handles);
  ASSERT_EQ(status, kTfLiteOk);

  // Second allocation with the same model in the arena:
  subgraph_allocations = micro_allocator->StartModelAllocation(model);
  ASSERT_NE(nullptr, subgraph_allocations);

  status = kTfLiteOk, micro_allocator->FinishModelAllocation(
                          model, subgraph_allocations, &scratch_buffer_handles);
  ASSERT_EQ(status, kTfLiteOk);

  size_t tensors_count = tflite::testing::GetModelTensorCount(model);

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kTfLiteEvalTensorData);

  // Node and tensor arrays must be allocated as well as each node and tensor.
  EXPECT_EQ(recorded_allocation.count, tensors_count * 2);
  EXPECT_EQ(recorded_allocation.requested_bytes,
            tensors_count * TF_LITE_EVAL_TENSOR_STRUCT_SIZE * 2);
  EXPECT_GE(recorded_allocation.used_bytes,
            tensors_count * TF_LITE_EVAL_TENSOR_STRUCT_SIZE * 2);
}

TEST(RecordingMicroAllocatorTest, TestRecordsPersistentTfLiteTensorData) {
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize);
  ASSERT_NE(micro_allocator, nullptr);

  TfLiteTensor* tensor = micro_allocator->AllocatePersistentTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, 0, 0);
  ASSERT_NE(tensor, nullptr);

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kPersistentTfLiteTensorData);

  EXPECT_EQ(recorded_allocation.count, static_cast<size_t>(1));
  EXPECT_EQ(recorded_allocation.requested_bytes, TF_LITE_TENSOR_STRUCT_SIZE);
  EXPECT_GE(recorded_allocation.used_bytes, TF_LITE_TENSOR_STRUCT_SIZE);
}

TEST(RecordingMicroAllocatorTest,
     TestRecordsPersistentTfLiteTensorQuantizationData) {
  const tflite::Model* model = tflite::GetModel(kTestConvModelData);
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize);
  ASSERT_NE(micro_allocator, nullptr);

  TfLiteTensor* tensor = micro_allocator->AllocatePersistentTfLiteTensor(
      model, /*subgraph_allocations=*/nullptr, 0, 0);
  ASSERT_NE(tensor, nullptr);

  // Walk the model subgraph to find all tensors with quantization params and
  // keep a tally.
  size_t quantized_channel_bytes = 0;
  const tflite::Tensor* cur_tensor =
      model->subgraphs()->Get(0)->tensors()->Get(0);
  const tflite::QuantizationParameters* quantization_params =
      cur_tensor->quantization();
  if (quantization_params && quantization_params->scale() &&
      quantization_params->scale()->size() > 0 &&
      quantization_params->zero_point() &&
      quantization_params->zero_point()->size() > 0) {
    size_t num_channels = quantization_params->scale()->size();
    quantized_channel_bytes += TfLiteIntArrayGetSizeInBytes(num_channels);
  }

  // Calculate the expected allocation bytes with subgraph quantization data:
  size_t expected_requested_bytes =
      TF_LITE_AFFINE_QUANTIZATION_SIZE + quantized_channel_bytes;

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::
              kPersistentTfLiteTensorQuantizationData);

  // Each quantized tensors has 2 mallocs (quant struct, zero point dimensions):
  EXPECT_EQ(recorded_allocation.count, static_cast<size_t>(2));
  EXPECT_EQ(recorded_allocation.requested_bytes, expected_requested_bytes);
  EXPECT_GE(recorded_allocation.used_bytes, expected_requested_bytes);
}

TEST(RecordingMicroAllocatorTest, TestRecordsPersistentBufferData) {
  uint8_t arena[kTestConvArenaSize];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, kTestConvArenaSize);
  ASSERT_NE(micro_allocator, nullptr);

  void* buffer = micro_allocator->AllocatePersistentBuffer(/*bytes=*/100);
  ASSERT_NE(buffer, nullptr);

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kPersistentBufferData);

  EXPECT_EQ(recorded_allocation.count, static_cast<size_t>(1));
  EXPECT_EQ(recorded_allocation.requested_bytes, static_cast<size_t>(100));
  EXPECT_GE(recorded_allocation.used_bytes, static_cast<size_t>(100));

  buffer = micro_allocator->AllocatePersistentBuffer(/*bytes=*/50);
  ASSERT_NE(buffer, nullptr);

  recorded_allocation = micro_allocator->GetRecordedAllocation(
      tflite::RecordedAllocationType::kPersistentBufferData);

  EXPECT_EQ(recorded_allocation.count, static_cast<size_t>(2));
  EXPECT_EQ(recorded_allocation.requested_bytes, static_cast<size_t>(150));
  EXPECT_GE(recorded_allocation.used_bytes, static_cast<size_t>(150));
}

TEST(RecordingMicroAllocatorTest, TestMultiSubgraphModel) {
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver ops_resolver;
  const tflite::Model* model =
      tflite::testing::GetSimpleModelWithNullInputsAndOutputs();
  const int arena_size = 2048;

  uint8_t arena[arena_size];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, arena_size);
  ASSERT_NE(micro_allocator, nullptr);

  tflite::SubgraphAllocations* subgraph_allocations =
      micro_allocator->StartModelAllocation(model);
  ASSERT_NE(nullptr, subgraph_allocations);

  TfLiteStatus status = micro_allocator->FinishModelAllocation(
      model, subgraph_allocations, &scratch_buffer_handles);
  ASSERT_EQ(status, kTfLiteOk);

  size_t num_ops = 0;
  size_t num_tensors = 0;
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
       subgraph_idx++) {
    const tflite::SubGraph* subgraph = model->subgraphs()->Get(subgraph_idx);
    num_ops += subgraph->operators()->size();
    num_tensors += subgraph->tensors()->size();
  }

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kNodeAndRegistrationArray);

  EXPECT_EQ(recorded_allocation.count, num_ops);
  EXPECT_EQ(recorded_allocation.requested_bytes,
            num_ops * NODE_AND_REGISTRATION_STRUCT_SIZE);
  EXPECT_GE(recorded_allocation.used_bytes,
            num_ops * NODE_AND_REGISTRATION_STRUCT_SIZE);

  recorded_allocation = micro_allocator->GetRecordedAllocation(
      tflite::RecordedAllocationType::kTfLiteEvalTensorData);

  EXPECT_EQ(recorded_allocation.count, num_tensors);
  EXPECT_EQ(recorded_allocation.requested_bytes,
            num_tensors * TF_LITE_EVAL_TENSOR_STRUCT_SIZE);
  EXPECT_GE(recorded_allocation.used_bytes,
            num_tensors * TF_LITE_EVAL_TENSOR_STRUCT_SIZE);
}

#ifdef USE_TFLM_COMPRESSION

TEST(RecordingMicroAllocatorTest, TestCompressedModel) {
  tflite::ScratchBufferHandle* scratch_buffer_handles = nullptr;
  tflite::testing::TestingOpResolver ops_resolver;
  const tflite::Model* model = tflite::testing::GetSimpleMockModelCompressed();
  const int arena_size = 2048;

  uint8_t arena[arena_size];

  tflite::RecordingMicroAllocator* micro_allocator =
      tflite::RecordingMicroAllocator::Create(arena, arena_size);
  EXPECT_NE(micro_allocator, nullptr);

  tflite::SubgraphAllocations* subgraph_allocations =
      micro_allocator->StartModelAllocation(model);
  EXPECT_NE(nullptr, subgraph_allocations);

  TfLiteStatus status = micro_allocator->FinishModelAllocation(
      model, subgraph_allocations, &scratch_buffer_handles);
  ASSERT_EQ(status, kTfLiteOk);

  micro_allocator->PrintAllocations();

  size_t count_compression_allocations = 0;
  size_t size_compression_allocations = 0;
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
       subgraph_idx++) {
    const tflite::CompressionTensorData** ctl =
        subgraph_allocations[subgraph_idx].compressed.tensors;
    if (ctl == nullptr) {
      continue;
    }
    const tflite::SubGraph* subgraph = model->subgraphs()->Get(subgraph_idx);
    const size_t num_tensors = subgraph->tensors()->size();
    for (size_t i = 0; i < num_tensors; i++) {
      if (ctl[i] != nullptr) {
        count_compression_allocations++;
        size_compression_allocations += sizeof(tflite::CompressionTensorData);
        count_compression_allocations++;
        size_compression_allocations += sizeof(tflite::LookupTableData);
      }
    }
    // Add the CompressionTensorData array
    count_compression_allocations++;
    size_compression_allocations +=
        num_tensors * sizeof(tflite::CompressionTensorData*);
  }

  tflite::RecordedAllocation recorded_allocation =
      micro_allocator->GetRecordedAllocation(
          tflite::RecordedAllocationType::kCompressionData);

  EXPECT_EQ(recorded_allocation.count, count_compression_allocations);
  EXPECT_EQ(recorded_allocation.requested_bytes, size_compression_allocations);
  EXPECT_GE(recorded_allocation.used_bytes, size_compression_allocations);
}

#endif  // USE_TFLM_COMPRESSION

// TODO(b/158124094): Find a way to audit OpData allocations on
// cross-architectures.

TF_LITE_MICRO_TESTS_MAIN
