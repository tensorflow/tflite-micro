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

#include "tensorflow/lite/micro/tools/benchmarking/metrics.h"

#include <sys/types.h>

#include <cstddef>

#include "tensorflow/lite/micro/tools/benchmarking/log_utils.h"

namespace tflite {

void LogArenaAllocations(
    const tflite::RecordingSingleArenaBufferAllocator* allocator,
    const PrettyPrintType type) {
  constexpr int kArenaRows = 3;
  constexpr int kArenaCols = 3;

  const size_t total_bytes = allocator->GetUsedBytes();

  size_t allocations[kArenaRows] = {total_bytes,
                                    allocator->GetNonPersistentUsedBytes(),
                                    allocator->GetPersistentUsedBytes()};
  char titles[kArenaRows][kMaxStringLength] = {"Total", "Head", "Tail"};
  char headers[kArenaRows][kMaxStringLength] = {"Arena", "Bytes", "% Arena"};

  char data[kArenaCols][kArenaRows][kMaxStringLength];
  for (int i = 0; i < kArenaRows; ++i) {
    MicroStrcpy(data[0][i], titles[i]);
    FormatNumber<int32_t>(data[1][i], allocations[i]);
    FormatAsPercentage(data[2][i], static_cast<int64_t>(allocations[i]),
                       static_cast<int64_t>(total_bytes), 2);
  }

  PrintFormattedData<kArenaRows, kArenaCols>(headers, data, kArenaRows, type,
                                             "Arena");
}

void LogAllocations(const tflite::RecordingMicroAllocator& allocator,
                    const PrettyPrintType type) {
  constexpr int kAllocationTypes = 7;
  tflite::RecordedAllocationType types[kAllocationTypes] = {
      tflite::RecordedAllocationType::kTfLiteEvalTensorData,
      tflite::RecordedAllocationType::kPersistentTfLiteTensorData,
      tflite::RecordedAllocationType::kPersistentTfLiteTensorQuantizationData,
      tflite::RecordedAllocationType::kPersistentBufferData,
      tflite::RecordedAllocationType::kTfLiteTensorVariableBufferData,
      tflite::RecordedAllocationType::kNodeAndRegistrationArray,
      tflite::RecordedAllocationType::kOpData};

  char titles[kAllocationTypes][kMaxStringLength] = {
      "Eval tensor data",
      "Persistent tensor data",
      "Persistent quantization data",
      "Persistent buffer data",
      "Tensor variable buffer data",
      "Node and registration array",
      "Operation data"};

  constexpr int kColumns = 6;
  const char headers[kColumns][kMaxStringLength] = {
      "Allocation", "Id", "Used", "Requested", "Count", "% Memory"};

  const size_t total_bytes =
      allocator.GetSimpleMemoryAllocator()->GetUsedBytes();

  char data[kColumns][kAllocationTypes][kMaxStringLength];
  for (int i = 0; i < kAllocationTypes; ++i) {
    tflite::RecordedAllocation allocation =
        allocator.GetRecordedAllocation(types[i]);
    MicroStrcpy(data[0][i], titles[i]);
    FormatNumber<int32_t>(data[1][i], static_cast<int>(types[i]));
    FormatNumber<int32_t>(data[2][i], allocation.used_bytes);
    FormatNumber<int32_t>(data[3][i], allocation.requested_bytes);
    FormatNumber<int32_t>(data[4][i], allocation.count);
    FormatAsPercentage(data[5][i], static_cast<int64_t>(allocation.used_bytes),
                       static_cast<int64_t>(total_bytes), 2);
  }

  PrintFormattedData<kAllocationTypes, kColumns>(
      headers, data, kAllocationTypes, type, "Allocations");
}

void LogAllocatorEvents(const tflite::RecordingMicroAllocator& allocator,
                        const PrettyPrintType type) {
  LogArenaAllocations(allocator.GetSimpleMemoryAllocator(), type);
  LogAllocations(allocator, type);
}
}  // namespace tflite
