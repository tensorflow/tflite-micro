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

namespace tflite {

MicroContext::MicroContext(MicroAllocator* allocator, const Model* model,
                           MicroGraph* graph)
    : allocator_(allocator), model_(model), graph_(graph) {}

void* MicroContext::AllocatePersistentBuffer(TfLiteContext* ctx, size_t bytes) {
  return GetMicroContext(ctx)->allocator_->AllocatePersistentBuffer(bytes);
}

TfLiteStatus MicroContext::RequestScratchBufferInArena(TfLiteContext* context,
                                                       size_t bytes,
                                                       int* buffer_idx) {
  MicroContext* micro_context = GetMicroContext(context);
  return micro_context->allocator_->RequestScratchBufferInArena(
      bytes, micro_context->graph_->GetCurrentSubgraphIndex(), buffer_idx);
}

void* MicroContext::GetScratchBuffer(TfLiteContext* context, int buffer_idx) {
  ScratchBufferHandle* handle =
      GetMicroContext(context)->scratch_buffer_handles_ + buffer_idx;
  return handle->data;
}
void MicroContext::ReportOpError(struct TfLiteContext* context,
                                 const char* format, ...) {
  va_list args;
  va_start(args, format);
  MicroPrintf(format, args);
  va_end(args);
}

MicroGraph* MicroContext::GetGraph() { return graph_; }

TfLiteTensor* MicroContext::GetTensor(const struct TfLiteContext* context,
                                      int tensor_idx) {
  MicroContext* micro_context = GetMicroContext(context);
  return micro_context->allocator_->AllocateTempTfLiteTensor(
      micro_context->model_, micro_context->graph_->GetAllocations(),
      tensor_idx, micro_context->graph_->GetCurrentSubgraphIndex());
}

TfLiteEvalTensor* MicroContext::GetEvalTensor(
    const struct TfLiteContext* context, int tensor_idx) {
  MicroContext* micro_context = GetMicroContext(context);
  return &micro_context->graph_
              ->GetAllocations()[micro_context->graph_
                                     ->GetCurrentSubgraphIndex()]
              .tensors[tensor_idx];
}

MicroContext* MicroContext::GetMicroContext(
    const struct TfLiteContext* context) {
  return reinterpret_cast<MicroContext*>(context->impl_);
}

void MicroContext::SetScratchBufferHandles(
    ScratchBufferHandle* scratch_buffer_handles) {
  scratch_buffer_handles_ = scratch_buffer_handles;
}

TfLiteStatus MicroContext::SetExternalContext(void* external_context_payload) {
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

TfLiteExternalContext* MicroContext::GetExternalContext(
    TfLiteContext* context, TfLiteExternalContextType unused) {
  return reinterpret_cast<TfLiteExternalContext*>(
      GetMicroContext(context)->external_context_payload_);
}

}  // namespace tflite
