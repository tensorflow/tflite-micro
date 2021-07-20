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

#ifndef XCORE_OP_UTILS_H_
#define XCORE_OP_UTILS_H_

#include <cassert>
#include <cstdint>
#include <utility>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_utils.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "xcore_persistent_array.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

/* Get size (in bytes) given a TfLiteType enum
 *  This is useful because a tensor's type field is a TfLiteType
 *
 *  Returns kTfLiteError if the type is not supported
 *
 *  NOTE: This is cribbed from tensorflow/lite/util.h because TFLu does not
 * fully support the methods defined in tensorflow/lite/util.h
 */
TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes);

template <typename T>
static inline T* construct_persistent_object(TfLiteContext* context) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return new (context->AllocatePersistentBuffer(context, sizeof(T))) T;
}

static inline TfLiteStatus request_scratch_if_needed(TfLiteContext* context,
                                                     const void* source_address,
                                                     const size_t size,
                                                     int& scratch_idx) {
  if (source_address && !is_ram_address((uintptr_t)source_address)) {
    return context->RequestScratchBufferInArena(context, size, &scratch_idx);
  }
  return kTfLiteOk;
}

static inline TfLiteStatus request_scratch_if_needed(TfLiteContext* context,
                                                     const TfLiteTensor* tensor,
                                                     int& scratch_idx) {
  return request_scratch_if_needed(context, tensor->data.data, tensor->bytes,
                                   scratch_idx);
}

template <typename T>
static inline TfLiteStatus fetch_scratch_if_needed(
    TfLiteContext* context, T*& array, const TfLiteEvalTensor* tensor,
    int scratch_idx) {
  if (scratch_idx >= 0) {
    array =
        static_cast<const T*>(context->GetScratchBuffer(context, scratch_idx));
    const RuntimeShape shape = tflite::micro::GetTensorShape(tensor);

    size_t sizeof_tensor_type;
    GetSizeOfType(context, tensor->type, &sizeof_tensor_type);

    FetchBuffer((int8_t**)&array, tflite::micro::GetTensorData<int8_t>(tensor),
                shape.FlatSize() * sizeof_tensor_type);
  } else {
    array = tflite::micro::GetTensorData<T>(tensor);
  }
  TF_LITE_ENSURE(context, array);
  return kTfLiteOk;
}

template <typename TArgs, typename TThreadData>
struct MultiThreadedOpData {
  TArgs args;
  PersistentArray<TThreadData> threads;
  int stack_scratch_index = -1;
  size_t stack_size;
};

#ifndef UNSUPPORTED_KERNEL_TYPE
#define UNSUPPORTED_KERNEL_TYPE(T) \
  TF_LITE_MICRO_FAIL("Unsupported " #T " value")
#endif /*UNSUPPORTED_KERNEL_TYPE*/

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OP_UTILS_H_
