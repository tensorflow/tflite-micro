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
#ifndef TENSORFLOW_LITE_MICRO_C_C_API_TYPES_H_
#define TENSORFLOW_LITE_MICRO_C_C_API_TYPES_H_

#include <stdint.h>

#ifdef __cplusplus
namespace tflite {
namespace micro {
#endif

typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10,
  kTfLiteFloat64 = 11,
  kTfLiteComplex128 = 12,
  kTfLiteUInt64 = 13,
  kTfLiteResource = 14,
  kTfLiteVariant = 15,
  kTfLiteUInt32 = 16,
  kTfLiteUInt16 = 17,
  kTfLiteInt4 = 18,
  kTfLiteBFloat16 = 19,
  kTfLiteInt2 = 20,
  kTfLiteUInt4 = 21,
  kTfLiteFloat8E4M3FN = 22,
  kTfLiteFloat8E5M2 = 23,
} TfLiteType;

typedef struct TfLiteQuantizationParams {
  float scale;
  int32_t zero_point;
} TfLiteQuantizationParams;

typedef enum TfLiteStatus {
  kTfLiteOk = 0,
  kTfLiteError = 1,
  kTfLiteDelegateError = 2,
  kTfLiteApplicationError = 3,
  kTfLiteDelegateDataNotFound = 4,
  kTfLiteDelegateDataWriteError = 5,
  kTfLiteDelegateDataReadError = 6,
  kTfLiteUnresolvedOps = 7,
  kTfLiteCancelled = 8,
  kTfLiteOutputShapeNotKnown = 9,
  kTfLiteAbort = 15,
} TfLiteStatus;

#ifdef __cplusplus
}  // namespace micro

using micro::kTfLiteAbort;
using micro::kTfLiteApplicationError;
using micro::kTfLiteBFloat16;
using micro::kTfLiteBool;
using micro::kTfLiteCancelled;
using micro::kTfLiteComplex128;
using micro::kTfLiteComplex64;
using micro::kTfLiteDelegateDataNotFound;
using micro::kTfLiteDelegateDataReadError;
using micro::kTfLiteDelegateDataWriteError;
using micro::kTfLiteDelegateError;
using micro::kTfLiteError;
using micro::kTfLiteFloat16;
using micro::kTfLiteFloat32;
using micro::kTfLiteFloat64;
using micro::kTfLiteFloat8E4M3FN;
using micro::kTfLiteFloat8E5M2;
using micro::kTfLiteInt16;
using micro::kTfLiteInt2;
using micro::kTfLiteInt32;
using micro::kTfLiteInt4;
using micro::kTfLiteInt64;
using micro::kTfLiteInt8;
using micro::kTfLiteNoType;
using micro::kTfLiteOk;
using micro::kTfLiteOutputShapeNotKnown;
using micro::kTfLiteResource;
using micro::kTfLiteString;
using micro::kTfLiteUInt16;
using micro::kTfLiteUInt32;
using micro::kTfLiteUInt4;
using micro::kTfLiteUInt64;
using micro::kTfLiteUInt8;
using micro::kTfLiteUnresolvedOps;
using micro::kTfLiteVariant;
using micro::TfLiteQuantizationParams;
using micro::TfLiteStatus;
using micro::TfLiteType;

}  // namespace tflite

#if !defined(TFLM_NO_GLOBAL_C_ALIASES) &&              \
    !defined(TENSORFLOW_LITE_CORE_C_C_API_TYPES_H_) && \
    !defined(TENSORFLOW_LITE_C_C_API_TYPES_H_) &&      \
    !defined(TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_TFLITE_TYPES_H_)
using ::tflite::micro::kTfLiteAbort;
using ::tflite::micro::kTfLiteApplicationError;
using ::tflite::micro::kTfLiteBFloat16;
using ::tflite::micro::kTfLiteBool;
using ::tflite::micro::kTfLiteCancelled;
using ::tflite::micro::kTfLiteComplex128;
using ::tflite::micro::kTfLiteComplex64;
using ::tflite::micro::kTfLiteDelegateDataNotFound;
using ::tflite::micro::kTfLiteDelegateDataReadError;
using ::tflite::micro::kTfLiteDelegateDataWriteError;
using ::tflite::micro::kTfLiteDelegateError;
using ::tflite::micro::kTfLiteError;
using ::tflite::micro::kTfLiteFloat16;
using ::tflite::micro::kTfLiteFloat32;
using ::tflite::micro::kTfLiteFloat64;
using ::tflite::micro::kTfLiteFloat8E4M3FN;
using ::tflite::micro::kTfLiteFloat8E5M2;
using ::tflite::micro::kTfLiteInt16;
using ::tflite::micro::kTfLiteInt2;
using ::tflite::micro::kTfLiteInt32;
using ::tflite::micro::kTfLiteInt4;
using ::tflite::micro::kTfLiteInt64;
using ::tflite::micro::kTfLiteInt8;
using ::tflite::micro::kTfLiteNoType;
using ::tflite::micro::kTfLiteOk;
using ::tflite::micro::kTfLiteOutputShapeNotKnown;
using ::tflite::micro::kTfLiteResource;
using ::tflite::micro::kTfLiteString;
using ::tflite::micro::kTfLiteUInt16;
using ::tflite::micro::kTfLiteUInt32;
using ::tflite::micro::kTfLiteUInt4;
using ::tflite::micro::kTfLiteUInt64;
using ::tflite::micro::kTfLiteUInt8;
using ::tflite::micro::kTfLiteUnresolvedOps;
using ::tflite::micro::kTfLiteVariant;
using ::tflite::micro::TfLiteQuantizationParams;
using ::tflite::micro::TfLiteStatus;
using ::tflite::micro::TfLiteType;
#endif
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_MICRO_C_C_API_TYPES_H_
