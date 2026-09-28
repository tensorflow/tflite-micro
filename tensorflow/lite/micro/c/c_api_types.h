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
extern "C" {
#endif

#if !defined(TENSORFLOW_LITE_CORE_C_C_API_TYPES_H_) && \
    !defined(TENSORFLOW_LITE_C_C_API_TYPES_H_) &&      \
    !defined(TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_TFLITE_TYPES_H_)
#define TENSORFLOW_LITE_CORE_C_C_API_TYPES_H_
#define TENSORFLOW_LITE_C_C_API_TYPES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_TFLITE_TYPES_H_

#ifdef SWIG
#define TFL_CAPI_EXPORT
#elif defined(TFL_STATIC_LIBRARY_BUILD)
#define TFL_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TFL_COMPILE_LIBRARY
#define TFL_CAPI_EXPORT __declspec(dllexport)
#else
#define TFL_CAPI_EXPORT
#endif
#else
#define TFL_CAPI_EXPORT __attribute__((visibility("default")))
#endif
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

typedef enum TfLiteDimensionType {
  kTfLiteDimDense = 0,
  kTfLiteDimSparseCSR,
} TfLiteDimensionType;

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

typedef struct TfLiteOpaqueContext TfLiteOpaqueContext;
typedef struct TfLiteOpaqueNode TfLiteOpaqueNode;
typedef struct TfLiteOpaqueTensor TfLiteOpaqueTensor;
typedef struct TfLiteDelegate TfLiteDelegate;
typedef struct TfLiteOpaqueDelegateStruct TfLiteOpaqueDelegateStruct;
typedef struct TfLiteDelegate TfLiteOpaqueDelegate;

#endif  // !defined(TENSORFLOW_LITE_CORE_C_C_API_TYPES_H_)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_MICRO_C_C_API_TYPES_H_
