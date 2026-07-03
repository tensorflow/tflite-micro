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
#include <limits>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

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
  switch (type) {
    case kTfLiteFloat16:
      *size = sizeof(int16_t);
      break;
    case kTfLiteBFloat16:
      *size = sizeof(int16_t);
      break;
    case kTfLiteFloat32:
      *size = sizeof(float);
      break;
    case kTfLiteFloat64:
      *size = sizeof(double);
      break;
    case kTfLiteInt16:
      *size = sizeof(int16_t);
      break;
    case kTfLiteInt32:
      *size = sizeof(int32_t);
      break;
    case kTfLiteUInt32:
      *size = sizeof(uint32_t);
      break;
    case kTfLiteUInt8:
      *size = sizeof(uint8_t);
      break;
    case kTfLiteUInt16:
      *size = sizeof(uint16_t);
      break;
    case kTfLiteInt8:
      *size = sizeof(int8_t);
      break;
    case kTfLiteInt64:
      *size = sizeof(int64_t);
      break;
    case kTfLiteUInt64:
      *size = sizeof(uint64_t);
      break;
    case kTfLiteBool:
      *size = sizeof(bool);
      break;
    case kTfLiteResource:
      *size = sizeof(int32_t);
      break;
    case kTfLiteComplex64:
      *size = sizeof(float) * 2;
      break;
    case kTfLiteComplex128:
      *size = sizeof(double) * 2;
      break;
    case kTfLiteInt4:
      *size = sizeof(int8_t);
      break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus BytesRequiredForTensor(const tflite::Tensor& flatbuffer_tensor,
                                    size_t* bytes, size_t* type_size) {
  // Shape dimensions and the element type size come from the partially-trusted
  // flatbuffer model, so an oversized shape (e.g. [65536, 65536]) is an invalid
  // model topology that -- per the Error Handling & Defensive Programming Guide
  // -- must be rejected during the Setup phase (this runs under MicroAllocator).
  // The running product is done in size_t (matching the result type) and every
  // multiplication is guarded against wrap-around. This helper is not given a
  // TfLiteContext, so the guide's low-cost `return kTfLiteError;` branch is used
  // rather than TF_LITE_ENSURE.
  size_t element_count = 1;
  // If flatbuffer_tensor.shape == nullptr, then flatbuffer_tensor is a scalar
  // so has 1 element.
  if (flatbuffer_tensor.shape() != nullptr) {
    for (size_t n = 0; n < flatbuffer_tensor.shape()->size(); ++n) {
      const int32_t dim = flatbuffer_tensor.shape()->Get(n);
      if (dim < 0) {
        return kTfLiteError;
      }
      const size_t udim = static_cast<size_t>(dim);
      if (udim != 0 &&
          element_count > std::numeric_limits<size_t>::max() / udim) {
        return kTfLiteError;
      }
      element_count *= udim;
    }
  }

  TfLiteType tf_lite_type;
  TF_LITE_ENSURE_STATUS(
      ConvertTensorType(flatbuffer_tensor.type(), &tf_lite_type));
  TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(tf_lite_type, type_size));
  if (*type_size != 0 &&
      element_count > std::numeric_limits<size_t>::max() / *type_size) {
    return kTfLiteError;
  }
  *bytes = element_count * (*type_size);
  return kTfLiteOk;
}

TfLiteStatus TfLiteEvalTensorByteLength(const TfLiteEvalTensor* eval_tensor,
                                        size_t* out_bytes) {
  TFLITE_DCHECK(out_bytes != nullptr);

  // Called from both the Setup planning path and from Eval helpers (e.g.
  // squeeze, kernel_util). The running product is computed in size_t and each
  // multiplication is guarded against wrap-around. Because there is no
  // TfLiteContext here and the guard protects against runtime memory
  // corruption, the guide's low-cost `return kTfLiteError;` branch is used.
  size_t element_count = 1;
  // If eval_tensor->dims == nullptr, then tensor is a scalar so has 1 element.
  if (eval_tensor->dims != nullptr) {
    for (int n = 0; n < eval_tensor->dims->size; ++n) {
      const int dim = eval_tensor->dims->data[n];
      if (dim < 0) {
        return kTfLiteError;
      }
      const size_t udim = static_cast<size_t>(dim);
      if (udim != 0 &&
          element_count > std::numeric_limits<size_t>::max() / udim) {
        return kTfLiteError;
      }
      element_count *= udim;
    }
  }
  size_t type_size;
  TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(eval_tensor->type, &type_size));
  if (type_size != 0 &&
      element_count > std::numeric_limits<size_t>::max() / type_size) {
    return kTfLiteError;
  }
  *out_bytes = element_count * type_size;
  return kTfLiteOk;
}

TfLiteStatus AllocateOutputDimensionsFromInput(TfLiteContext* context,
                                               const TfLiteTensor* input1,
                                               const TfLiteTensor* input2,
                                               TfLiteTensor* output) {
  const TfLiteTensor* input = nullptr;
  TF_LITE_ENSURE(context, input1->dims != nullptr);
  TF_LITE_ENSURE(context, input2->dims != nullptr);
  TF_LITE_ENSURE(context, output->dims->size == 0);

  input = input1->dims->size > input2->dims->size ? input1 : input2;
  TF_LITE_ENSURE(context, output->type == input->type);
  // `bytes` is the tensor storage size (element size * shape product) that is
  // stored in output->bytes. Keep it distinct from `dimensions_count`: the
  // output->dims array is sized from the number of dimension entries, not from
  // the byte count.
  size_t bytes = 0;
  TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(input->type, &bytes));
  const int dimensions_count = tflite::GetTensorShape(input).DimensionsCount();
  for (int i = 0; i < dimensions_count; i++) {
    const int dim = input->dims->data[i];
    const size_t udim = static_cast<size_t>(dim);
    // This helper runs in the Setup/Prepare phase, so per the Error Handling &
    // Defensive Programming Guide a bad (partially-trusted model) shape is
    // rejected with TF_LITE_ENSURE. The two preconditions -- a non-negative
    // dimension and a running product that does not overflow size_t -- are
    // combined into a single TF_LITE_ENSURE so only one error string is emitted
    // into .rodata per call site (see "The Hidden Cost of TF_LITE_ENSURE").
    // The `dim >= 0` term is evaluated first, so the division guard is never
    // reached with a wrapped-around `udim`.
    TF_LITE_ENSURE(
        context,
        dim >= 0 &&
            (udim == 0 || bytes <= std::numeric_limits<size_t>::max() / udim));
    bytes *= udim;
  }
  output->bytes = bytes;

  output->dims =
      reinterpret_cast<TfLiteIntArray*>(context->AllocatePersistentBuffer(
          context, TfLiteIntArrayGetSizeInBytes(dimensions_count)));
  output->dims->size = input->dims->size;
  for (int i = 0; i < dimensions_count; i++) {
    output->dims->data[i] = input->dims->data[i];
  }
  return kTfLiteOk;
}

}  // namespace tflite
