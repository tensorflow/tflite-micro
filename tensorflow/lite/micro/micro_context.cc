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

#include "tensorflow/lite/micro/micro_context.h"

#include <algorithm>
#include <cstddef>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace micro {
namespace {

int GetTensorIndex(int index, int max_size, const int* tensor_indices) {
  if (index >= 0 && index < max_size) {
    const int tensor_index = tensor_indices[index];
    if (tensor_index != kTfLiteOptionalTensor) {
      return tensor_index;
    }
  }
  return -1;
}

}  // namespace

TfLiteTensor* MicroContext::AllocateTempInputTensor(const TfLiteNode* node,
                                                    int index) {
  const int tensor_index =
      GetTensorIndex(index, node->inputs->size, node->inputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

TfLiteTensor* MicroContext::AllocateTempOutputTensor(const TfLiteNode* node,
                                                     int index) {
  const int tensor_index =
      GetTensorIndex(index, node->outputs->size, node->outputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

TfLiteTensor* MicroContext::AllocateTempIntermediateTensor(
    const TfLiteNode* node, int index) {
  const int tensor_index = GetTensorIndex(index, node->intermediates->size,
                                          node->intermediates->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

TfLiteStatus MicroContext::SetDecompressionMemory(
    const AlternateMemoryRegion* regions, size_t count) {
  if (decompress_regions_ != nullptr) {
    return kTfLiteError;
  }

  decompress_regions_ = regions;
  decompress_regions_size_ = count;
  decompress_regions_allocations_ = static_cast<size_t*>(
      AllocatePersistentBuffer(sizeof(size_t) * decompress_regions_size_));
  if (decompress_regions_allocations_ == nullptr) {
    return kTfLiteError;
  }
  ResetDecompressionMemoryAllocations();

  return kTfLiteOk;
}

void* MicroContext::AllocateDecompressionMemory(size_t bytes,
                                                size_t alignment) {
  if (decompress_regions_ != nullptr) {
    for (size_t i = 0; i < decompress_regions_size_; i++) {
      const AlternateMemoryRegion* region = &decompress_regions_[i];
      uint8_t* start = static_cast<uint8_t*>(region->address) +
                       decompress_regions_allocations_[i];
      uint8_t* aligned_start = AlignPointerUp(start, alignment);
      size_t total = bytes + (aligned_start - start);
      if (total + decompress_regions_allocations_[i] <= region->bytes) {
        decompress_regions_allocations_[i] += total;
        return aligned_start;
      }
    }
  }

  return nullptr;
}

void MicroContext::ResetDecompressionMemoryAllocations() {
  if (decompress_regions_ == nullptr) {
    return;
  }
  TFLITE_DCHECK(decompress_regions_allocations_ != nullptr);
  std::fill_n(decompress_regions_allocations_, decompress_regions_size_, 0);
}

TfLiteStatus MicroContext::SetCustomDecodeRegistrations(
    const CustomDecodeRegistration* registrations, size_t count) {
  if (custom_decode_registrations_ != nullptr) {
    return kTfLiteError;
  }
  custom_decode_registrations_ = registrations;
  custom_decode_registrations_size_ = count;
  return kTfLiteOk;
}

void MicroContext::InitTfLiteContext(TfLiteContext* context) {
  context->impl_ = static_cast<void*>(this);
}

}  // namespace micro
}  // namespace tflite
