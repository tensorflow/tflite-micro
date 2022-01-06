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

#include "tensorflow/lite/micro/fake_micro_context.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"

namespace tflite {
using ::tflite::MicroArenaBufferAlignment;

FakeMicroContext::FakeMicroContext(TfLiteTensor* tensors,
                                   SimpleMemoryAllocator* allocator,
                                   MicroGraph* micro_graph)
    : MicroContext(nullptr, nullptr, nullptr),
      tensors_(tensors),
      allocator_(allocator),
      micro_graph_(micro_graph) {}

FakeMicroContext* FakeMicroContext::GetMicroContext(
    const struct TfLiteContext* context) {
  return reinterpret_cast<FakeMicroContext*>(context->impl_);
}

TfLiteTensor* FakeMicroContext::GetTensor(const struct TfLiteContext* context,
                                          int tensor_index) {
  TFLITE_DCHECK(context != nullptr);

  FakeMicroContext* mock_context = GetMicroContext(context);
  TFLITE_DCHECK(mock_context != nullptr);
  return &mock_context->tensors_[tensor_index];
}

TfLiteEvalTensor* FakeMicroContext::GetEvalTensor(
    const struct TfLiteContext* context, int tensor_index) {
  TFLITE_DCHECK(context != nullptr);
  FakeMicroContext* runner = GetMicroContext(context);
  TFLITE_DCHECK(runner != nullptr);

  TfLiteEvalTensor* eval_tensor =
      reinterpret_cast<TfLiteEvalTensor*>(runner->allocator_->AllocateTemp(
          sizeof(TfLiteEvalTensor), alignof(TfLiteEvalTensor)));
  TFLITE_DCHECK(eval_tensor != nullptr);

  // In unit tests, the TfLiteTensor pointer contains the source of truth for
  // buffers and values:
  eval_tensor->data = runner->tensors_[tensor_index].data;
  eval_tensor->dims = runner->tensors_[tensor_index].dims;
  eval_tensor->type = runner->tensors_[tensor_index].type;
  return eval_tensor;
}

void* FakeMicroContext::AllocatePersistentBuffer(size_t bytes) {
  return allocator_->AllocateFromTail(bytes, MicroArenaBufferAlignment());
}

TfLiteStatus FakeMicroContext::RequestScratchBufferInArena(size_t bytes,
                                                           int* buffer_index) {
  TFLITE_DCHECK(buffer_index != nullptr);

  if (scratch_buffer_count_ == kNumScratchBuffers_) {
    MicroPrintf("Exceeded the maximum number of scratch tensors allowed (%d).",
                kNumScratchBuffers_);
    return kTfLiteError;
  }

  // For tests, we allocate scratch buffers from the tail and keep them around
  // for the lifetime of model. This means that the arena size in the tests will
  // be more than what we would have if the scratch buffers could share memory.
  scratch_buffers_[scratch_buffer_count_] =
      allocator_->AllocateFromTail(bytes, MicroArenaBufferAlignment());
  TFLITE_DCHECK(scratch_buffers_[scratch_buffer_count_] != nullptr);

  *buffer_index = scratch_buffer_count_++;
  return kTfLiteOk;
}

void* FakeMicroContext::GetScratchBuffer(TfLiteContext* context,
                                         int buffer_index) {
  TFLITE_DCHECK(context != nullptr);
  FakeMicroContext* runner = GetMicroContext(context);
  TFLITE_DCHECK(runner != nullptr);

  TFLITE_DCHECK(runner->scratch_buffer_count_ <= kNumScratchBuffers_);
  if (buffer_index >= runner->scratch_buffer_count_) {
    return nullptr;
  }
  return runner->scratch_buffers_[buffer_index];
}

void FakeMicroContext::ReportOpError(struct TfLiteContext* context,
                                     const char* format, ...) {
  va_list args;
  va_start(args, format);
  GetMicroErrorReporter()->Report(format, args);
  va_end(args);
}

MicroGraph* FakeMicroContext::GetGraph() { return micro_graph_; }

}  // namespace tflite
