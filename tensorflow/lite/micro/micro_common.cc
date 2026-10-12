/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/micro_common.h"

#include <cstdarg>
#include <cstring>

#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

template <class T>
size_t TfLiteVarArrayGetSizeInBytes(const int size) {
  constexpr size_t data_size = sizeof(decltype(((T*)nullptr)->data[0]));
  size_t computed_size = sizeof(T) + data_size * size;
#if defined(_MSC_VER)
  computed_size -= data_size;
#endif
  return computed_size;
}

template <class T, class U>
int TfLiteVarArrayEqualsArray(const T* const a, const int b_size,
                              const U* const b_data) {
  if (a == nullptr) {
    return b_size == 0;
  }
  if (a->size != b_size) {
    return 0;
  }
  return !memcmp(a->data, b_data, a->size * sizeof(a->data[0]));
}

template <class T>
int TfLiteVarArrayEqual(const T* const a, const T* const b) {
  if (a == b) {
    return 1;
  }
  if (a == nullptr || b == nullptr) {
    return 0;
  }
  return TfLiteVarArrayEqualsArray(a, b->size, b->data);
}

}  // namespace

const char* TfLiteTypeGetName(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return "NOTYPE";
    case kTfLiteFloat32:
      return "FLOAT32";
    case kTfLiteUInt16:
      return "UINT16";
    case kTfLiteInt16:
      return "INT16";
    case kTfLiteInt32:
      return "INT32";
    case kTfLiteUInt32:
      return "UINT32";
    case kTfLiteUInt8:
      return "UINT8";
    case kTfLiteInt8:
      return "INT8";
    case kTfLiteInt64:
      return "INT64";
    case kTfLiteUInt64:
      return "UINT64";
    case kTfLiteBool:
      return "BOOL";
    case kTfLiteComplex64:
      return "COMPLEX64";
    case kTfLiteComplex128:
      return "COMPLEX128";
    case kTfLiteString:
      return "STRING";
    case kTfLiteFloat16:
      return "FLOAT16";
    case kTfLiteFloat64:
      return "FLOAT64";
    case kTfLiteResource:
      return "RESOURCE";
    case kTfLiteVariant:
      return "VARIANT";
    case kTfLiteInt4:
      return "INT4";
    case kTfLiteBFloat16:
      return "BFLOAT16";
    case kTfLiteInt2:
      return "INT2";
    case kTfLiteUInt4:
      return "UINT4";
    case kTfLiteFloat8E4M3FN:
      return "FLOAT8E4M3FN";
    case kTfLiteFloat8E5M2:
      return "FLOAT8E5M2";
  }
  return "Unknown type";
}

int TfLiteTypeGetSize(TfLiteType type) {
  int size_bits = TfLiteTypeGetSizeBits(type);
  if (size_bits % 8 == 0) {
    return size_bits / 8;
  } else {
    // For non-byte sized types, return 0.
    return 0;
  }
}

int TfLiteTypeGetSizeBits(TfLiteType type) {
  switch (type) {
    case kTfLiteInt2:
      return 2;
    case kTfLiteInt4:
    case kTfLiteUInt4:
      return 4;
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteFloat8E4M3FN:
    case kTfLiteFloat8E5M2:
      return 8;
    case kTfLiteUInt16:
    case kTfLiteInt16:
    case kTfLiteFloat16:
    case kTfLiteBFloat16:
      return 16;
    case kTfLiteFloat32:
    case kTfLiteInt32:
    case kTfLiteUInt32:
      return 32;
    case kTfLiteInt64:
    case kTfLiteUInt64:
    case kTfLiteFloat64:
    case kTfLiteComplex64:
      return 64;
    case kTfLiteComplex128:
      return 128;
    case kTfLiteBool:
      return sizeof(bool) * 8;
    case kTfLiteString:
    case kTfLiteNoType:
    case kTfLiteResource:
    case kTfLiteVariant:
      break;
  }
  return 0;
}

size_t TfLiteIntArrayGetSizeInBytes(int size) {
  return TfLiteVarArrayGetSizeInBytes<TfLiteIntArray>(size);
}

int TfLiteIntArrayEqual(const TfLiteIntArray* a, const TfLiteIntArray* b) {
  return TfLiteVarArrayEqual(a, b);
}

int TfLiteIntArrayEqualsArray(const TfLiteIntArray* a, int b_size,
                              const int b_data[]) {
  return TfLiteVarArrayEqualsArray(a, b_size, b_data);
}

void* MicroContextAllocatePersistentBuffer(TfLiteContext* ctx, size_t bytes) {
  return GetMicroContext(ctx)->AllocatePersistentBuffer(bytes);
}

TfLiteStatus MicroContextRequestScratchBufferInArena(TfLiteContext* ctx,
                                                     size_t bytes,
                                                     int* buffer_idx) {
  return GetMicroContext(ctx)->RequestScratchBufferInArena(bytes, buffer_idx);
}

void* MicroContextGetScratchBuffer(TfLiteContext* ctx, int buffer_idx) {
  return GetMicroContext(ctx)->GetScratchBuffer(buffer_idx);
}

TfLiteEvalTensor* MicroContextGetEvalTensor(const TfLiteContext* context,
                                            int tensor_idx) {
  return GetMicroContext(context)->GetEvalTensor(tensor_idx);
}

void MicroContextReportOpError(TfLiteContext* context, const char* format,
                               ...) {
  va_list args;
  va_start(args, format);
  VMicroPrintf(format, args);
  va_end(args);
}

}  // namespace micro
}  // namespace tflite
