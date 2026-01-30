/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/linear_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/micro_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocation_info.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifdef USE_TFLM_COMPRESSION
#include <algorithm>
#include <cstring>
#include "tensorflow/lite/micro/compression/metadata_saved.h"
#endif  // USE_TFLM_COMPRESSION

namespace tflite {
namespace {

constexpr size_t kMaxScratchBuffersPerOp = 12;
constexpr int kUnassignedScratchBufferRequestIndex = -1;
const TfLiteIntArray kZeroLengthIntArray = {};

class MicroBuiltinDataAllocator : public TfLiteBridgeBuiltinDataAllocator {
 public:
  explicit MicroBuiltinDataAllocator(
      IPersistentBufferAllocator* persistent_allocator)
      : persistent_allocator_(persistent_allocator) {}

  void* Allocate(size_t size, size_t alignment_hint) override {
    return persistent_allocator_->AllocatePersistentBuffer(size, alignment_hint);
  }
  void Deallocate(void* data) override {}

 private:
  IPersistentBufferAllocator* persistent_allocator_;
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

MicroMemoryPlanner* CreateMemoryPlanner(
    MemoryPlannerType memory_planner_type,
    IPersistentBufferAllocator* memory_allocator) {
  MicroMemoryPlanner* memory_planner = nullptr;
  uint8_t* memory_planner_buffer = nullptr;

  switch (memory_planner_type) {
    case MemoryPlannerType::kLinear: {
      memory_planner_buffer = memory_allocator->AllocatePersistentBuffer(
          sizeof(LinearMemoryPlanner), alignof(LinearMemoryPlanner));
      memory_planner = new (memory_planner_buffer) LinearMemoryPlanner();
      break;
    }
    case MemoryPlannerType::kGreedy: {
      memory_planner_buffer = memory_allocator->AllocatePersistentBuffer(
          sizeof(GreedyMemoryPlanner), alignof(GreedyMemoryPlanner));
      memory_planner = new (memory_planner_buffer) GreedyMemoryPlanner();
      break;
    }
  }
  return memory_planner;
}

TfLiteStatus CreatePlan(MicroMemoryPlanner* planner,
                        const AllocationInfo* allocation_info,
                        size_t allocation_info_size) {
  for (size_t i = 0; i < allocation_info_size; ++i) {
    const AllocationInfo* current = &allocation_info[i];
    if (current->needs_allocating) {
      size_t aligned_bytes_required =
          AlignSizeUp(current->bytes, MicroArenaBufferAlignment());
      if (current->offline_offset == kOnlinePlannedBuffer) {
        TF_LITE_ENSURE_STATUS(planner->AddBuffer(aligned_bytes_required,
                                                 current->first_created,
                                                 current->last_used));
      } else {
        TF_LITE_ENSURE_STATUS(
            planner->AddBuffer(aligned_bytes_required, current->first_created,
                               current->last_used, current->offline_offset));
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus CommitPlan(MicroMemoryPlanner* planner, uint8_t* starting_point,
                        const AllocationInfo* allocation_info,
                        size_t allocation_info_size) {
  int planner_index = 0;
  for (size_t i = 0; i < allocation_info_size; ++i) {
    const AllocationInfo* current = &allocation_info[i];
    if (current->needs_allocating) {
      int offset = -1;
      TF_LITE_ENSURE_STATUS(
          planner->GetOffsetForBuffer(planner_index, &offset));
      *current->output_ptr = reinterpret_cast<void*>(starting_point + offset);
      ++planner_index;
    }
  }
  return kTfLiteOk;
}

IPersistentBufferAllocator* CreatePersistentArenaAllocator(uint8_t* buffer_head,
                                                           size_t buffer_size) {
  uint8_t* aligned_buffer_tail =
      AlignPointerDown(buffer_head + buffer_size, MicroArenaBufferAlignment());
  size_t aligned_buffer_size = aligned_buffer_tail - buffer_head;
  PersistentArenaBufferAllocator tmp =
      PersistentArenaBufferAllocator(buffer_head, aligned_buffer_size);

  uint8_t* allocator_buffer =
      tmp.AllocatePersistentBuffer(sizeof(PersistentArenaBufferAllocator),
                                   alignof(PersistentArenaBufferAllocator));
  return new (allocator_buffer) PersistentArenaBufferAllocator(tmp);
}

INonPersistentBufferAllocator* CreateNonPersistentArenaAllocator(
    uint8_t* buffer_head, size_t buffer_size,
    IPersistentBufferAllocator* persistent_buffer_allocator) {
  uint8_t* allocator_buffer =
      persistent_buffer_allocator->AllocatePersistentBuffer(
          sizeof(NonPersistentArenaBufferAllocator),
          alignof(NonPersistentArenaBufferAllocator));
  uint8_t* aligned_buffer_head =
      AlignPointerUp(buffer_head, MicroArenaBufferAlignment());
  size_t aligned_buffer_size = buffer_head + buffer_size - aligned_buffer_head;

  INonPersistentBufferAllocator* non_persistent_buffer_allocator =
      new (allocator_buffer) NonPersistentArenaBufferAllocator(
          aligned_buffer_head, aligned_buffer_size);
  return non_persistent_buffer_allocator;
}

}  // namespace

namespace internal {

// --- التعديل يبدأ هنا (The Fix) ---
void* GetFlatbufferTensorBuffer(
    const tflite::Tensor& flatbuffer_tensor,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers) {
  
  if (buffers == nullptr) return nullptr;

  // استخراج الـ index
  const int32_t buffer_index = flatbuffer_tensor.buffer();

  // فحص الأمان: التأكد من أن الـ index داخل حدود المصفوفة
  if (buffer_index < 0 || static_cast<size_t>(buffer_index) >= buffers->size()) {
    #ifndef TF_LITE_STRIP_ERROR_STRINGS
    MicroPrintf("Security Alert: Tensor buffer index %d is out of bounds!", buffer_index);
    #endif
    return nullptr; 
  }

  void* out_buffer = nullptr;
  if (auto* buffer = (*buffers)[buffer_index]) {
    if (auto* array = buffer->data()) {
      if (array->size()) {
        out_buffer = const_cast<void*>(static_cast<const void*>(array->data()));
      }
    }
  }
  return out_buffer;
}
// --- التعديل ينتهي هنا ---

TfLiteStatus InitializeTfLiteTensorFromFlatbuffer(
    IPersistentBufferAllocator* persistent_buffer_allocator,
    INonPersistentBufferAllocator* non_persistent_buffer_allocator,
    bool allocate_temp, const tflite::Tensor& flatbuffer_tensor,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    TfLiteTensor* result) {
  TFLITE_DCHECK(result != nullptr);
  *result = {};
  TF_LITE_ENSURE_STATUS(
      tflite::ConvertTensorType(flatbuffer_tensor.type(), &result->type));
  result->is_variable = flatbuffer_tensor.is_variable();

  result->data.data = GetFlatbufferTensorBuffer(flatbuffer_tensor, buffers);

  if (result->data.data == nullptr) {
    result->allocation_type = kTfLiteArenaRw;
  } else {
    result->allocation_type = kTfLiteMmapRo;
  }

  size_t type_size;
  TF_LITE_ENSURE_STATUS(
      BytesRequiredForTensor(flatbuffer_tensor, &result->bytes, &type_size));

  if (flatbuffer_tensor.shape() == nullptr) {
    result->dims = const_cast<TfLiteIntArray*>(&kZeroLengthIntArray);
  } else {
    result->dims = FlatBufferVectorToTfLiteTypeArray(flatbuffer_tensor.shape());
  }

  const auto* src_quantization = flatbuffer_tensor.quantization();
  if (src_quantization && src_quantization->scale() &&
      (src_quantization->scale()->size() > 0) &&
      src_quantization->zero_point() &&
      (src_quantization->zero_point()->size() > 0)) {
    result->params.scale = src_quantization->scale()->Get(0);
    result->params.zero_point =
        static_cast<int32_t>(src_quantization->zero_point()->Get(0));

    int channels = src_quantization->scale()->size();
    TfLiteAffineQuantization* quantization =
        allocate_temp
            ? reinterpret_cast<TfLiteAffineQuantization*>(
                  non_persistent_buffer_allocator->AllocateTemp(
                      sizeof(TfLiteAffineQuantization),
                      alignof(TfLiteAffineQuantization)))
            : reinterpret_cast<TfLiteAffineQuantization*>(
                  persistent_buffer_allocator->AllocatePersistentBuffer(
                      sizeof(TfLiteAffineQuantization),
                      alignof(TfLiteAffineQuantization)));
    if (quantization == nullptr) {
      MicroPrintf("Unable to allocate TfLiteAffineQuantization.\n");
      return kTfLiteError;
    }

    quantization->zero_point =
        allocate_temp
            ? reinterpret_cast<TfLiteIntArray*>(
                  non_persistent_buffer_allocator->AllocateTemp(
                      TfLiteIntArrayGetSizeInBytes(channels),
                      alignof(TfLiteIntArray)))
            : reinterpret_cast<TfLiteIntArray*>(
                  persistent_buffer_allocator->AllocatePersistentBuffer(
                      TfLiteIntArrayGetSizeInBytes(channels),
                      alignof(TfLiteIntArray)));
    if (quantization->zero_point == nullptr) {
      MicroPrintf("Unable to allocate quantization->zero_point.\n");
      return kTfLiteError;
    }

    quantization->scale =
        FlatBufferVectorToTfLiteTypeArray(src_quantization->scale());

    quantization->zero_point->size = channels;
    int* zero_point_data = quantization->zero_point->data;
    for (int i = 0; i < channels; i++) {
      zero_point_data[i] = src_quantization->zero_point()->size() ==
                                   src_quantization->scale()->size()
                               ? src_quantization->zero_point()->Get(i)
                               : src_quantization->zero_point()->Get(0);
    }
    quantization->quantized_dimension = src_quantization->quantized_dimension();
    result->quantization = {kTfLiteAffineQuantization, quantization};
  }
  return kTfLiteOk;
}

TfLiteStatus InitializeTfLiteEvalTensorFromFlatbuffer(
    const tflite::Tensor& flatbuffer_tensor,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    TfLiteEvalTensor* result) {
  *result = {};
  TF_LITE_ENSURE_STATUS(
      tflite::ConvertTensorType(flatbuffer_tensor.type(), &result->type));

  result->data.data = GetFlatbufferTensorBuffer(flatbuffer_tensor, buffers);

  if (flatbuffer_tensor.shape() == nullptr) {
    result->dims = const_cast<TfLiteIntArray*>(&kZeroLengthIntArray);
  } else {
    result->dims = FlatBufferVectorToTfLiteTypeArray(flatbuffer_tensor.shape());
  }
  return kTfLiteOk;
}

// ... باقي الكود يبقى كما هو دون تغيير ...
#ifdef USE_TFLM_COMPRESSION
// (دوال الـ Compression تبقى كما هي)
#endif

}  // namespace internal

// ... (بقية دوال MicroAllocator تبقى كما هي) ...

}  // namespace tflite
