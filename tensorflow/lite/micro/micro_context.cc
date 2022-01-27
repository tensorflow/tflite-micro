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

#include "tensorflow/lite/micro/micro_context.h"

#include <cstdarg>
#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace tflite {
MicroContext::MicroContext(MicroAllocator* allocator, const Model* model,
                           MicroGraph* graph)
    : allocator_(*allocator), graph_(*graph), model_(model) {}

MicroContext::~MicroContext() {}

void* MicroContext::AllocatePersistentBuffer(size_t bytes) {
  return allocator_.AllocatePersistentBuffer(bytes);
}

TfLiteStatus MicroContext::RequestScratchBufferInArena(size_t bytes,
                                                       int* buffer_idx) {
  return allocator_.RequestScratchBufferInArena(
      graph_.GetAllocations(), bytes, graph_.GetCurrentSubgraphIndex(),
      buffer_idx);
}

void* MicroContext::GetScratchBuffer(int buffer_idx) {
  SubgraphAllocations subgraph_allocations =
      graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()];
  ScratchBufferHandle* handle =
      subgraph_allocations.scratch_buffer_handles + buffer_idx;
  return handle->data;
}

TfLiteTensor* MicroContext::GetTensor(int tensor_idx) {
  return allocator_.AllocateTempTfLiteTensor(model_, graph_.GetAllocations(),
                                             tensor_idx,
                                             graph_.GetCurrentSubgraphIndex());
}

TfLiteEvalTensor* MicroContext::GetEvalTensor(int tensor_idx) {
  return &graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()]
              .tensors[tensor_idx];
}

TfLiteStatus MicroContext::set_external_context(
    void* external_context_payload) {
  if (external_context_payload == nullptr ||
      external_context_payload_ != nullptr) {
    MicroPrintf(
        "Attempting to set external context to %x but it was %x already",
        external_context_payload, external_context_payload_);
    return kTfLiteError;
  }

  external_context_payload_ = external_context_payload;
  return kTfLiteOk;
}

void MicroContextReportOpError(struct TfLiteContext* context,
                               const char* format, ...) {
  va_list args;
  va_start(args, format);
  GetMicroErrorReporter()->Report(format, args);
  va_end(args);
}

}  // namespace tflite
