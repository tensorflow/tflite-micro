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

#include "tensorflow/lite/micro/micro_interpreter_context.h"

#include <cstdint>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

namespace {

#ifdef USE_TFLM_COMPRESSION

int GetInputTensorIndex(const TfLiteNode* node, const int index) {
  if (index >= 0 && index < node->inputs->size) {
    const int tensor_index = node->inputs->data[index];
    if (tensor_index != kTfLiteOptionalTensor) {
      return tensor_index;
    }
  }
  return -1;
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace

MicroInterpreterContext::MicroInterpreterContext(MicroAllocator* allocator,
                                                 const Model* model,
                                                 MicroInterpreterGraph* graph)
    : allocator_(*allocator),
      graph_(*graph),
      model_(model),
      state_(InterpreterState::kInit) {}

MicroInterpreterContext::~MicroInterpreterContext() {}

void* MicroInterpreterContext::AllocatePersistentBuffer(size_t bytes) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInit);
  return allocator_.AllocatePersistentBuffer(bytes);
}

TfLiteStatus MicroInterpreterContext::RequestScratchBufferInArena(
    size_t bytes, int* buffer_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);
  return allocator_.RequestScratchBufferInArena(
      bytes, graph_.GetCurrentSubgraphIndex(), buffer_idx);
}

void* MicroInterpreterContext::GetScratchBuffer(int buffer_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kInvoke);
  ScratchBufferHandle* handle = scratch_buffer_handles_ + buffer_idx;
  return handle->data;
}

TfLiteTensor* MicroInterpreterContext::AllocateTempTfLiteTensor(
    int tensor_idx) {
  return allocator_.AllocateTempTfLiteTensor(model_, graph_.GetAllocations(),
                                             tensor_idx,
                                             graph_.GetCurrentSubgraphIndex());
}

void MicroInterpreterContext::DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
  return allocator_.DeallocateTempTfLiteTensor(tensor);
}

uint8_t* MicroInterpreterContext::AllocateTempBuffer(size_t size,
                                                     size_t alignment) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);
  return allocator_.AllocateTempBuffer(size, alignment);
}

void MicroInterpreterContext::DeallocateTempBuffer(uint8_t* buffer) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);
  allocator_.DeallocateTempBuffer(buffer);
}

TfLiteEvalTensor* MicroInterpreterContext::GetEvalTensor(int tensor_idx) {
  return &graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()]
              .tensors[tensor_idx];
}

void MicroInterpreterContext::SetScratchBufferHandles(
    ScratchBufferHandle* scratch_buffer_handles) {
  scratch_buffer_handles_ = scratch_buffer_handles;
}

TfLiteStatus MicroInterpreterContext::set_external_context(
    void* external_context_payload) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInvoke);
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

void MicroInterpreterContext::SetInterpreterState(InterpreterState state) {
  state_ = state;
}

MicroInterpreterContext::InterpreterState
MicroInterpreterContext::GetInterpreterState() const {
  return state_;
}

#ifdef USE_TFLM_COMPRESSION

// Available during Prepare & Eval. Returns false if tensor is not
// compressed.
bool MicroInterpreterContext::IsTensorCompressed(const TfLiteNode* node,
                                                 int tensor_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInvoke);

  const SubgraphAllocations* allocations =
      &graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()];
  if (allocations->compressed.tensors == nullptr) {
    return false;
  }
  int index = GetInputTensorIndex(node, tensor_idx);
  if (index == -1) {
    return false;
  }
  return allocations->compressed.tensors[index] != nullptr;
}

// Only available during Prepare. The kernel is responsible for storing the
// scratch buffer handle.
int MicroInterpreterContext::AllocateDecompressionScratchBuffer(
    const TfLiteNode* node, int tensor_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);

  const SubgraphAllocations* allocations =
      &graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()];
  if (allocations->compressed.tensors == nullptr) {
    return -1;
  }
  int index = GetInputTensorIndex(node, tensor_idx);
  if (index == -1 || allocations->compressed.tensors[index] == nullptr) {
    return -1;
  }
  const TfLiteEvalTensor* tensor = &allocations->tensors[index];
  const size_t byte_count = EvalTensorBytes(tensor);
  int scratch_index = -1;
  TfLiteStatus result = RequestScratchBufferInArena(byte_count, &scratch_index);
  if (result != kTfLiteOk) {
    return -1;
  }

  return scratch_index;
}

// Available during Prepare & Eval. Returns nullptr if tensor is not
// compressed.
const CompressionTensorData* MicroInterpreterContext::GetTensorCompressionData(
    const TfLiteNode* node, int tensor_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInvoke);

  const SubgraphAllocations* allocations =
      &graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()];
  if (allocations->compressed.tensors == nullptr) {
    return nullptr;
  }
  int index = GetInputTensorIndex(node, tensor_idx);
  if (index == -1) {
    return nullptr;
  }
  return allocations->compressed.tensors[index];
}

// Only available during Eval. Returns nullptr on failure, otherwise returns a
// pointer to the scratch buffer.
void* MicroInterpreterContext::DecompressTensorToScratchBuffer(
    const TfLiteEvalTensor& tensor,
    const CompressionTensorData& compression_data, int scratch_buffer_handle) {
  TFLITE_DCHECK(state_ == InterpreterState::kInvoke);

  return MicroContext::DecompressTensorToScratchBuffer(tensor, compression_data,
                                                       scratch_buffer_handle);
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace tflite
