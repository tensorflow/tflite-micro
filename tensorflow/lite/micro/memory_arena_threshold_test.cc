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

#include <stdint.h>

#include "tensorflow/lite/micro/kernels/svdf.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/models/keyword_scrambled_model_data.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_conv_model.h"

/**
 * Tests to ensure arena memory allocation does not regress by more than 3%.
 */

namespace {

// Ensure memory doesn't expand more that 3%:
constexpr float kAllocationThreshold = 0.03;

// TODO(b/160617245): Record persistent allocations to provide a more accurate
// number here.
constexpr float kAllocationTailMiscCeiling = 2 * 1024;

const bool kIs64BitSystem = (sizeof(void*) == 8);

constexpr int kKeywordModelTensorArenaSize = 22 * 1024;
uint8_t keyword_model_tensor_arena[kKeywordModelTensorArenaSize];

constexpr int kKeywordModelTensorCount = 54;
constexpr int kKeywordModelNodeAndRegistrationCount = 15;

// NOTE: These values are measured on x86-64:
// TODO(b/158651472): Consider auditing these values on non-64 bit systems.
// TODO(b/199414774): use expression for hardcoded constants such as
// kKeywordModelTotalSize.
//
// Run this test with '--copt=-DTF_LITE_STATIC_MEMORY' to get optimized memory
// runtime values:
#ifdef TF_LITE_STATIC_MEMORY
// Total size contributed by the keyword model excluding the
// RecordingMicroAllocator's overhead
// TODO(b/207157610): replace magic number that depends on OPs
constexpr int kKeywordModelOnlyTotalSize = 14472;
// Tail size contributed by the kdyword model excluding the
// RecordingMicroAllocator's overhead
// TODO(b/207157610): replace magic number that depends on OPs
constexpr int kKeywordModelOnlyTailSize = 13800;
constexpr int kKeywordModelPersistentTfLiteTensorDataSize = 128;
#else
// Total size contributed by the keyword model excluding the
// RecordingMicroAllocator's overhead.
// TODO(b/207157610): replace magic number that depends on OPs
constexpr int kKeywordModelOnlyTotalSize = 14936;
// Tail size contributed by the keyword model excluding the
// RecordingMicroAllocator's overhead
// TODO(b/207157610): replace magic number that depends on OPs
constexpr int kKeywordModelOnlyTailSize = 14264;
constexpr int kKeywordModelPersistentTfLiteTensorDataSize = 224;
#endif
constexpr int kKeywordModelHeadSize = 672;
constexpr int kKeywordModelTfLiteTensorVariableBufferDataSize = 10240;
constexpr int kKeywordModelPersistentTfLiteTensorQuantizationData = 64;
constexpr int kKeywordModelOpRuntimeDataSize = 148;

constexpr int kTestConvModelArenaSize = 12 * 1024;
uint8_t test_conv_tensor_arena[kTestConvModelArenaSize];

constexpr int kTestConvModelTensorCount = 15;
constexpr int kTestConvModelNodeAndRegistrationCount = 7;

#if defined(USE_TFLM_COMPRESSION)
constexpr int kKeywordModelPersistentBufferDataSize = 920;
#else
constexpr int kKeywordModelPersistentBufferDataSize = 840;
#endif

// NOTE: These values are measured on x86-64:
// TODO(b/158651472): Consider auditing these values on non-64 bit systems.
#ifdef TF_LITE_STATIC_MEMORY
// Total size contributed by the conv model excluding the
// RecordingMicroAllocator's overhead
// TODO(b/207157610): replace magic number that depends on OPs
constexpr int kTestConvModelOnlyTotalSize = 9576;
// Tail size contributed by the conv model excluding the
// RecordingMicroAllocator's overhead
// TODO(b/207157610): replace magic number that depends on OPs
constexpr int kTestConvModelOnlyTailSize = 1832;
constexpr int kTestConvModelPersistentTfLiteTensorDataSize = 128;
constexpr int kTestConvModelPersistentBufferDataSize = 748;
#else
// Total size contributed by the conv model excluding the
// RecordingMicroAllocator's overhead
// TODO(b/207157610): replace magic number that depends on OPs
constexpr int kTestConvModelOnlyTotalSize = 9832;
// Tail size contributed by the conv model excluding the
// RecordingMicroAllocator's overhead
// TODO(b/207157610): replace magic number that depends on OPs
constexpr int kTestConvModelOnlyTailSize = 2088;
constexpr int kTestConvModelPersistentTfLiteTensorDataSize = 224;
constexpr int kTestConvModelPersistentBufferDataSize = 740;
#endif
constexpr int kTestConvModelHeadSize = 7744;
constexpr int kTestConvModelOpRuntimeDataSize = 136;
constexpr int kTestConvModelPersistentTfLiteTensorQuantizationData = 0;

struct ModelAllocationThresholds {
  size_t tensor_count = 0;
  size_t node_and_registration_count = 0;
  size_t total_alloc_size = 0;
  size_t head_alloc_size = 0;
  size_t tail_alloc_size = 0;
  size_t tensor_variable_buffer_data_size = 0;
  size_t persistent_tflite_tensor_data_size = 0;
  size_t persistent_tflite_tensor_quantization_data_size = 0;
  size_t op_runtime_data_size = 0;
  size_t persistent_buffer_data = 0;
};

void EnsureAllocatedSizeThreshold(const char* allocation_type, size_t actual,
                                  size_t expected) {
  // TODO(b/158651472): Better auditing of non-64 bit systems:
  if (kIs64BitSystem) {
    // 64-bit systems should check floor and ceiling to catch memory savings:
    TF_LITE_MICRO_EXPECT_NEAR(actual, expected,
                              expected * kAllocationThreshold);
  } else {
    // Non-64 bit systems should just expect allocation does not exceed the
    // ceiling:
    TF_LITE_MICRO_EXPECT_LE(actual, expected + expected * kAllocationThreshold);
  }
}

void ValidateModelAllocationThresholds(
    const tflite::RecordingMicroAllocator& allocator,
    const ModelAllocationThresholds& thresholds) {
  MicroPrintf("Overhead from RecordingMicroAllocator is %d",
              tflite::RecordingMicroAllocator::GetDefaultTailUsage());
  allocator.PrintAllocations();

  EnsureAllocatedSizeThreshold(
      "Total", allocator.GetSimpleMemoryAllocator()->GetUsedBytes(),
      thresholds.total_alloc_size);
  EnsureAllocatedSizeThreshold(
      "Head", allocator.GetSimpleMemoryAllocator()->GetNonPersistentUsedBytes(),
      thresholds.head_alloc_size);
  EnsureAllocatedSizeThreshold(
      "Tail", allocator.GetSimpleMemoryAllocator()->GetPersistentUsedBytes(),
      thresholds.tail_alloc_size);
  EnsureAllocatedSizeThreshold(
      "TfLiteEvalTensor",
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kTfLiteEvalTensorData)
          .used_bytes,
      sizeof(TfLiteEvalTensor) * thresholds.tensor_count);
  EnsureAllocatedSizeThreshold(
      "VariableBufferData",
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kTfLiteTensorVariableBufferData)
          .used_bytes,
      thresholds.tensor_variable_buffer_data_size);
  EnsureAllocatedSizeThreshold(
      "PersistentTfLiteTensor",
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kPersistentTfLiteTensorData)
          .used_bytes,
      thresholds.persistent_tflite_tensor_data_size);
  EnsureAllocatedSizeThreshold(
      "PersistentTfliteTensorQuantizationData",
      allocator
          .GetRecordedAllocation(tflite::RecordedAllocationType::
                                     kPersistentTfLiteTensorQuantizationData)
          .used_bytes,
      thresholds.persistent_tflite_tensor_quantization_data_size);
  EnsureAllocatedSizeThreshold(
      "PersistentBufferData",
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kPersistentBufferData)
          .used_bytes,
      thresholds.persistent_buffer_data);
  EnsureAllocatedSizeThreshold(
      "NodeAndRegistration",
      allocator
          .GetRecordedAllocation(
              tflite::RecordedAllocationType::kNodeAndRegistrationArray)
          .used_bytes,
      sizeof(tflite::NodeAndRegistration) *
          thresholds.node_and_registration_count);

  // Ensure tail allocation recording is not missing any large chunks:
  size_t tail_est_length = sizeof(TfLiteEvalTensor) * thresholds.tensor_count +
                           thresholds.tensor_variable_buffer_data_size +
                           sizeof(tflite::NodeAndRegistration) *
                               thresholds.node_and_registration_count +
                           thresholds.op_runtime_data_size;
  TF_LITE_MICRO_EXPECT_LE(thresholds.tail_alloc_size - tail_est_length,
                          kAllocationTailMiscCeiling);
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestKeywordModelMemoryThreshold) {
  tflite::MicroMutableOpResolver<4> op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(
      op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8()),
      kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(op_resolver.AddQuantize(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(
      op_resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8_INT16()), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(op_resolver.AddSvdf(tflite::Register_SVDF_INT8()),
                          kTfLiteOk);
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(g_keyword_scrambled_model_data), op_resolver,
      keyword_model_tensor_arena, kKeywordModelTensorArenaSize);

  interpreter.AllocateTensors();

  ModelAllocationThresholds thresholds;
  thresholds.tensor_count = kKeywordModelTensorCount;
  thresholds.node_and_registration_count =
      kKeywordModelNodeAndRegistrationCount;
  thresholds.total_alloc_size =
      kKeywordModelOnlyTotalSize +
      tflite::RecordingMicroAllocator::GetDefaultTailUsage();
  thresholds.head_alloc_size = kKeywordModelHeadSize;
  thresholds.tail_alloc_size =
      kKeywordModelOnlyTailSize +
      tflite::RecordingMicroAllocator::GetDefaultTailUsage();
  thresholds.tensor_variable_buffer_data_size =
      kKeywordModelTfLiteTensorVariableBufferDataSize;
  thresholds.op_runtime_data_size = kKeywordModelOpRuntimeDataSize;
  thresholds.persistent_buffer_data = kKeywordModelPersistentBufferDataSize;
  thresholds.persistent_tflite_tensor_data_size =
      kKeywordModelPersistentTfLiteTensorDataSize;
  thresholds.persistent_tflite_tensor_quantization_data_size =
      kKeywordModelPersistentTfLiteTensorQuantizationData;

  ValidateModelAllocationThresholds(interpreter.GetMicroAllocator(),
                                    thresholds);
}

TF_LITE_MICRO_TEST(TestConvModelMemoryThreshold) {
  tflite::MicroMutableOpResolver<6> op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(op_resolver.AddConv2D(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(op_resolver.AddQuantize(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(op_resolver.AddMaxPool2D(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(op_resolver.AddReshape(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(op_resolver.AddFullyConnected(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(op_resolver.AddDequantize(), kTfLiteOk);

  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(kTestConvModelData), op_resolver, test_conv_tensor_arena,
      kTestConvModelArenaSize);

  interpreter.AllocateTensors();

  ModelAllocationThresholds thresholds;
  thresholds.tensor_count = kTestConvModelTensorCount;
  thresholds.node_and_registration_count =
      kTestConvModelNodeAndRegistrationCount;
  thresholds.total_alloc_size =
      kTestConvModelOnlyTotalSize +
      tflite::RecordingMicroAllocator::GetDefaultTailUsage();
  thresholds.head_alloc_size = kTestConvModelHeadSize;
  thresholds.tail_alloc_size =
      kTestConvModelOnlyTailSize +
      tflite::RecordingMicroAllocator::GetDefaultTailUsage();
  thresholds.op_runtime_data_size = kTestConvModelOpRuntimeDataSize;
  thresholds.persistent_buffer_data = kTestConvModelPersistentBufferDataSize;
  thresholds.persistent_tflite_tensor_data_size =
      kTestConvModelPersistentTfLiteTensorDataSize;
  thresholds.persistent_tflite_tensor_quantization_data_size =
      kTestConvModelPersistentTfLiteTensorQuantizationData;

  ValidateModelAllocationThresholds(interpreter.GetMicroAllocator(),
                                    thresholds);
}

TF_LITE_MICRO_TESTS_END
