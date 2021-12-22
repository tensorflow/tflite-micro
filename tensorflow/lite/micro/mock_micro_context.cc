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

#include "tensorflow/lite/micro/mock_micro_context.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"

namespace tflite {
using ::tflite::MicroArenaBufferAlignment;

MockMicroContext::MockMicroContext(TfLiteTensor* tensors,
                                   SimpleMemoryAllocator* allocator,
                                   MockMicroGraph* mock_micro_graph)
    : MicroContext(nullptr, nullptr, mock_micro_graph),
      tensors_(tensors),
      allocator_(allocator),
      mock_micro_graph_(mock_micro_graph) {}

MockMicroContext* MockMicroContext::GetMicroContext(
    const struct TfLiteContext* context) {
  return reinterpret_cast<MockMicroContext*>(context->impl_);
}

TfLiteTensor* MockMicroContext::GetTensor(const struct TfLiteContext* context,
                                          int tensor_index) {
  TFLITE_DCHECK(context != nullptr);

  MockMicroContext* mock_context = GetMicroContext(context);
  TFLITE_DCHECK(mock_context != nullptr);
  return &mock_context->tensors_[tensor_index];
}

TfLiteEvalTensor* MockMicroContext::GetEvalTensor(
    const struct TfLiteContext* context, int tensor_index) {
  TFLITE_DCHECK(context != nullptr);
  MockMicroContext* runner = GetMicroContext(context);
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

void* MockMicroContext::AllocatePersistentBuffer(TfLiteContext* context,
                                                 size_t bytes) {
  TFLITE_DCHECK(context != nullptr);
  MockMicroContext* runner = GetMicroContext(context);
  TFLITE_DCHECK(runner != nullptr);
  return runner->allocator_->AllocateFromTail(bytes,
                                              MicroArenaBufferAlignment());
}

TfLiteStatus MockMicroContext::RequestScratchBufferInArena(
    TfLiteContext* context, size_t bytes, int* buffer_index) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(buffer_index != nullptr);

  MockMicroContext* runner = GetMicroContext(context);
  TFLITE_DCHECK(runner != nullptr);

  if (runner->scratch_buffer_count_ == kNumScratchBuffers_) {
    MicroPrintf("Exceeded the maximum number of scratch tensors allowed (%d).",
                kNumScratchBuffers_);
    return kTfLiteError;
  }

  // For tests, we allocate scratch buffers from the tail and keep them around
  // for the lifetime of model. This means that the arena size in the tests will
  // be more than what we would have if the scratch buffers could share memory.
  runner->scratch_buffers_[runner->scratch_buffer_count_] =
      runner->allocator_->AllocateFromTail(bytes, MicroArenaBufferAlignment());
  TFLITE_DCHECK(runner->scratch_buffers_[runner->scratch_buffer_count_] !=
                nullptr);

  *buffer_index = runner->scratch_buffer_count_++;
  return kTfLiteOk;
}

void* MockMicroContext::GetScratchBuffer(TfLiteContext* context,
                                         int buffer_index) {
  TFLITE_DCHECK(context != nullptr);
  MockMicroContext* runner = GetMicroContext(context);
  TFLITE_DCHECK(runner != nullptr);

  TFLITE_DCHECK(runner->scratch_buffer_count_ <= kNumScratchBuffers_);
  if (buffer_index >= runner->scratch_buffer_count_) {
    return nullptr;
  }
  return runner->scratch_buffers_[buffer_index];
}

void MockMicroContext::ReportOpError(struct TfLiteContext* context,
                                     const char* format, ...) {
  va_list args;
  va_start(args, format);
  GetMicroErrorReporter()->Report(format, args);
  va_end(args);
}

}  // namespace tflite
