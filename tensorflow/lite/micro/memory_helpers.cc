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

#include "tensorflow/lite/micro/memory_helpers.h"

#include <cstddef>
#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/micro/flatbuffer_conversions.h"
#include "tensorflow/lite/micro/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace micro {

uint8_t* AlignPointerUp(uint8_t* data, size_t alignment) {
  std::uintptr_t data_as_uintptr_t = reinterpret_cast<std::uintptr_t>(data);
  uint8_t* aligned_result = reinterpret_cast<uint8_t*>(
      ((data_as_uintptr_t + (alignment - 1)) / alignment) * alignment);
  return aligned_result;
}

uint8_t* AlignPointerDown(uint8_t* data, size_t alignment) {
  std::uintptr_t data_as_uintptr_t = reinterpret_cast<std::uintptr_t>(data);
  uint8_t* aligned_result =
      reinterpret_cast<uint8_t*>((data_as_uintptr_t / alignment) * alignment);
  return aligned_result;
}

size_t AlignSizeUp(size_t size, size_t alignment) {
  size_t aligned_size = (((size + (alignment - 1)) / alignment) * alignment);
  return aligned_size;
}

TfLiteStatus TfLiteTypeSizeOf(TfLiteType type, size_t* size) {
  if (type == kTfLiteResource) {
    *size = sizeof(int32_t);
    return kTfLiteOk;
  } else if (type == kTfLiteInt4) {
    *size = sizeof(int8_t);
    return kTfLiteOk;
  }
  int bytes = TfLiteTypeGetSize(type);
  if (bytes == 0) {
    return kTfLiteError;
  }
  *size = bytes;
  return kTfLiteOk;
}

TfLiteStatus BytesRequiredForTensor(const tflite::Tensor& flatbuffer_tensor,
                                    size_t* bytes, size_t* type_size) {
  int element_count = 1;
  // If flatbuffer_tensor.shape == nullptr, then flatbuffer_tensor is a scalar
  // so has 1 element.
  if (flatbuffer_tensor.shape() != nullptr) {
    for (size_t n = 0; n < flatbuffer_tensor.shape()->size(); ++n) {
      element_count *= flatbuffer_tensor.shape()->Get(n);
    }
  }

  TfLiteType tf_lite_type;
  TF_LITE_ENSURE_STATUS(
      ConvertTensorType(flatbuffer_tensor.type(), &tf_lite_type));
  TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(tf_lite_type, type_size));
  *bytes = element_count * (*type_size);
  return kTfLiteOk;
}

TfLiteStatus TfLiteEvalTensorByteLength(const TfLiteEvalTensor* eval_tensor,
                                        size_t* out_bytes) {
  TFLITE_DCHECK(out_bytes != nullptr);

  int element_count = 1;
  // If eval_tensor->dims == nullptr, then tensor is a scalar so has 1 element.
  if (eval_tensor->dims != nullptr) {
    for (int n = 0; n < eval_tensor->dims->size; ++n) {
      element_count *= eval_tensor->dims->data[n];
    }
  }
  size_t type_size;
  TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(eval_tensor->type, &type_size));
  *out_bytes = element_count * type_size;
  return kTfLiteOk;
}

}  // namespace micro
}  // namespace tflite
