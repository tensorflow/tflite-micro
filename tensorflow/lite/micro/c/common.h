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
#ifndef TENSORFLOW_LITE_MICRO_C_COMMON_H_
#define TENSORFLOW_LITE_MICRO_C_COMMON_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/micro/c/c_api_types.h"

#ifndef kTfLiteOptionalTensor
#define kTfLiteOptionalTensor (-1)
#endif

#ifndef TF_LITE_KERNEL_LOG
#ifndef TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_KERNEL_LOG(context, ...)            \
  do {                                              \
    (context)->ReportError((context), __VA_ARGS__); \
  } while (false)
#else
#define TF_LITE_KERNEL_LOG(context, ...)
#endif
#endif  // TF_LITE_KERNEL_LOG

#ifndef TF_LITE_MAYBE_KERNEL_LOG
#ifndef TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_MAYBE_KERNEL_LOG(context, ...)        \
  do {                                                \
    if ((context) != nullptr) {                       \
      (context)->ReportError((context), __VA_ARGS__); \
    }                                                 \
  } while (false)
#else
#define TF_LITE_MAYBE_KERNEL_LOG(context, ...)
#endif
#endif  // TF_LITE_MAYBE_KERNEL_LOG

#ifndef TF_LITE_ENSURE_STATUS
#define TF_LITE_ENSURE_STATUS(a) \
  do {                           \
    TfLiteStatus s = (a);        \
    if (s != kTfLiteOk) {        \
      return s;                  \
    }                            \
  } while (0)
#endif

#ifndef TF_LITE_ENSURE
#define TF_LITE_ENSURE(context, a)                                      \
  do {                                                                  \
    if (!(a)) {                                                         \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s was not true.", __FILE__, \
                         __LINE__, #a);                                 \
      return kTfLiteError;                                              \
    }                                                                   \
  } while (0)
#endif

#ifndef TF_LITE_ENSURE_MSG
#define TF_LITE_ENSURE_MSG(context, value, ...)                \
  do {                                                         \
    if (!(value)) {                                            \
      TF_LITE_KERNEL_LOG((context), __FILE__ " " __VA_ARGS__); \
      return kTfLiteError;                                     \
    }                                                          \
  } while (0)
#endif

#ifndef TF_LITE_ENSURE_OK
#define TF_LITE_ENSURE_OK(context, a) \
  do {                                \
    TfLiteStatus s = (a);             \
    if (s != kTfLiteOk) {             \
      return s;                       \
    }                                 \
  } while (0)
#endif

#ifndef TF_LITE_ENSURE_EQ
#define TF_LITE_ENSURE_EQ(context, a, b)                                   \
  do {                                                                     \
    if ((a) != (b)) {                                                      \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s != %s (%d != %d)", __FILE__, \
                         __LINE__, #a, #b, (a), (b));                      \
      return kTfLiteError;                                                 \
    }                                                                      \
  } while (0)
#endif

#ifndef TF_LITE_ENSURE_TYPES_EQ
#define TF_LITE_ENSURE_TYPES_EQ(context, a, b)                             \
  do {                                                                     \
    if ((a) != (b)) {                                                      \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s != %s (%s != %s)", __FILE__, \
                         __LINE__, #a, #b, TfLiteTypeGetName(a),           \
                         TfLiteTypeGetName(b));                            \
      return kTfLiteError;                                                 \
    }                                                                      \
  } while (0)
#endif

#ifndef TF_LITE_ENSURE_NEAR
#define TF_LITE_ENSURE_NEAR(context, a, b, epsilon)                          \
  do {                                                                       \
    auto delta = ((a) > (b)) ? ((a) - (b)) : ((b) - (a));                    \
    if (delta > epsilon) {                                                   \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s not near %s (%f != %f)",       \
                         __FILE__, __LINE__, #a, #b, static_cast<double>(a), \
                         static_cast<double>(b));                            \
      return kTfLiteError;                                                   \
    }                                                                        \
  } while (0)
#endif

#ifdef __cplusplus
namespace tflite {
namespace micro {
#endif

// Forward declarations
struct TfLiteContext;
typedef struct TfLiteContext TfLiteContext;
struct TfLiteNode;
typedef struct TfLiteNode TfLiteNode;

typedef struct TfLiteIntArray {
  int size;
#if defined(_MSC_VER)
  int data[1];
#elif (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
       __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                             \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
  int data[0];
#else
  int data[];
#endif
} TfLiteIntArray;

size_t TfLiteIntArrayGetSizeInBytes(int size);
int TfLiteIntArrayEqual(const TfLiteIntArray* a, const TfLiteIntArray* b);
int TfLiteIntArrayEqualsArray(const TfLiteIntArray* a, int b_size,
                              const int b_data[]);

typedef struct TfLiteFloatArray {
  int size;
#if defined(_MSC_VER)
  float data[1];
#elif (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
       __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                             \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
  float data[0];
#else
  float data[];
#endif
} TfLiteFloatArray;

typedef struct TfLiteComplex64 {
  float re, im;
} TfLiteComplex64;

typedef struct TfLiteComplex128 {
  double re, im;
} TfLiteComplex128;

typedef struct TfLiteFloat16 {
  uint16_t data;
} TfLiteFloat16;

typedef struct TfLiteBFloat16 {
  uint16_t data;
} TfLiteBFloat16;

const char* TfLiteTypeGetName(TfLiteType type);

typedef enum TfLiteQuantizationType {
  kTfLiteNoQuantization = 0,
  kTfLiteAffineQuantization = 1,
  kTfLiteBlockwiseQuantization = 2,
  kTfLiteMultiAxisQuantization = 3,
} TfLiteQuantizationType;

typedef struct TfLiteQuantization {
  TfLiteQuantizationType type;
  void* params;
} TfLiteQuantization;

typedef struct TfLiteAffineQuantization {
  TfLiteFloatArray* scale;
  TfLiteIntArray* zero_point;
  int32_t quantized_dimension;
} TfLiteAffineQuantization;

typedef struct TfLiteBlockwiseQuantization {
  int32_t scale;
  int32_t zero_point;
  int32_t blocksize;
  int32_t quantized_dimension;
} TfLiteBlockwiseQuantization;

typedef struct TfLiteMultiAxisQuantization {
  int32_t scales;
  int32_t zero_points;
  int32_t blocksize;
  TfLiteIntArray* quantized_dimensions;
} TfLiteMultiAxisQuantization;

typedef union TfLitePtrUnion {
  int32_t* i32;
  uint32_t* u32;
  int64_t* i64;
  uint64_t* u64;
  float* f;
  TfLiteFloat16* f16;
  TfLiteBFloat16* bf16;
  double* f64;
  char* raw;
  const char* raw_const;
  uint8_t* uint8;
  bool* b;
  int16_t* i16;
  uint16_t* ui16;
  TfLiteComplex64* c64;
  TfLiteComplex128* c128;
  int8_t* int8;
  void* data;
} TfLitePtrUnion;

typedef enum TfLiteAllocationType {
  kTfLiteMemNone = 0,
  kTfLiteMmapRo,
  kTfLiteArenaRw,
  kTfLiteArenaRwPersistent,
  kTfLiteDynamic,
  kTfLitePersistentRo,
  kTfLiteCustom,
  kTfLiteVariantObject,
  kTfLiteNonCpu,
} TfLiteAllocationType;

typedef enum TfLiteExternalContextType {
  kTfLiteEigenContext = 0,
  kTfLiteGemmLowpContext = 1,
  kTfLiteEdgeTpuContext = 2,
  kTfLiteCpuBackendContext = 3,
  kTfLiteLiteRtBufferContext = 4,
  kTfLiteMaxExternalContexts = 5
} TfLiteExternalContextType;

typedef struct TfLiteExternalContext {
  TfLiteExternalContextType type;
  TfLiteStatus (*Refresh)(TfLiteContext* context);
} TfLiteExternalContext;

typedef struct TfLiteTensor {
  TfLiteQuantization quantization;
  TfLiteQuantizationParams params;
  TfLitePtrUnion data;
  TfLiteIntArray* dims;
  size_t bytes;
  TfLiteType type;
  TfLiteAllocationType allocation_type;
  bool is_variable;
} TfLiteTensor;

typedef struct TfLiteEvalTensor {
  TfLitePtrUnion data;
  TfLiteIntArray* dims;
  TfLiteType type;
} TfLiteEvalTensor;

typedef struct TfLiteNode {
  TfLiteIntArray* inputs;
  TfLiteIntArray* outputs;
  TfLiteIntArray* intermediates;
  void* user_data;
  void* builtin_data;
  const void* custom_initial_data;
  int custom_initial_data_size;
} TfLiteNode;

// TFLMRegistration defines the API that TFLM kernels need to implement.
typedef struct TFLMRegistration {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
  void (*reset)(TfLiteContext* context, void* buffer);
  int32_t builtin_code;
  const char* custom_name;
} TFLMRegistration;

typedef struct TFLMInferenceRegistration {
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
  void (*reset)(TfLiteContext* context, void* buffer);
} TFLMInferenceRegistration;

typedef struct TfLiteContext {
  size_t tensors_size;
  TfLiteTensor* tensors;
  void* impl_;

  void (*ReportError)(TfLiteContext*, const char* msg, ...);

  TfLiteExternalContext* (*GetExternalContext)(TfLiteContext*,
                                               TfLiteExternalContextType);

  void* (*AllocatePersistentBuffer)(TfLiteContext* ctx, size_t bytes);

  TfLiteStatus (*RequestScratchBufferInArena)(TfLiteContext* ctx, size_t bytes,
                                              int* buffer_idx);

  void* (*GetScratchBuffer)(TfLiteContext* ctx, int buffer_idx);

  TfLiteTensor* (*GetTensor)(const TfLiteContext* context, int tensor_idx);

  TfLiteEvalTensor* (*GetEvalTensor)(const TfLiteContext* context,
                                     int tensor_idx);
} TfLiteContext;

#ifdef __cplusplus
}  // namespace micro

using micro::kTfLiteAffineQuantization;
using micro::kTfLiteArenaRw;
using micro::kTfLiteArenaRwPersistent;
using micro::kTfLiteBlockwiseQuantization;
using micro::kTfLiteCpuBackendContext;
using micro::kTfLiteCustom;
using micro::kTfLiteDynamic;
using micro::kTfLiteEdgeTpuContext;
using micro::kTfLiteEigenContext;
using micro::kTfLiteGemmLowpContext;
using micro::kTfLiteLiteRtBufferContext;
using micro::kTfLiteMaxExternalContexts;
using micro::kTfLiteMemNone;
using micro::kTfLiteMmapRo;
using micro::kTfLiteMultiAxisQuantization;
using micro::kTfLiteNonCpu;
using micro::kTfLiteNoQuantization;
using micro::kTfLitePersistentRo;
using micro::kTfLiteVariantObject;
using micro::TfLiteAffineQuantization;
using micro::TfLiteAllocationType;
using micro::TfLiteBFloat16;
using micro::TfLiteBlockwiseQuantization;
using micro::TfLiteComplex128;
using micro::TfLiteComplex64;
using micro::TfLiteContext;
using micro::TfLiteEvalTensor;
using micro::TfLiteExternalContext;
using micro::TfLiteExternalContextType;
using micro::TfLiteFloat16;
using micro::TfLiteFloatArray;
using micro::TfLiteIntArray;
using micro::TfLiteIntArrayEqual;
using micro::TfLiteIntArrayEqualsArray;
using micro::TfLiteIntArrayGetSizeInBytes;
using micro::TfLiteMultiAxisQuantization;
using micro::TfLiteNode;
using micro::TfLitePtrUnion;
using micro::TfLiteQuantization;
using micro::TfLiteQuantizationType;
using micro::TfLiteTensor;
using micro::TfLiteTypeGetName;
using micro::TFLMInferenceRegistration;
using micro::TFLMRegistration;

}  // namespace tflite

using ::tflite::micro::TFLMInferenceRegistration;
using ::tflite::micro::TFLMRegistration;

#if !defined(TFLM_NO_GLOBAL_C_ALIASES) &&         \
    !defined(TENSORFLOW_LITE_CORE_C_COMMON_H_) && \
    !defined(TENSORFLOW_LITE_C_COMMON_H_)
using ::tflite::micro::kTfLiteAffineQuantization;
using ::tflite::micro::kTfLiteArenaRw;
using ::tflite::micro::kTfLiteArenaRwPersistent;
using ::tflite::micro::kTfLiteBlockwiseQuantization;
using ::tflite::micro::kTfLiteCpuBackendContext;
using ::tflite::micro::kTfLiteCustom;
using ::tflite::micro::kTfLiteDynamic;
using ::tflite::micro::kTfLiteEdgeTpuContext;
using ::tflite::micro::kTfLiteEigenContext;
using ::tflite::micro::kTfLiteGemmLowpContext;
using ::tflite::micro::kTfLiteLiteRtBufferContext;
using ::tflite::micro::kTfLiteMaxExternalContexts;
using ::tflite::micro::kTfLiteMemNone;
using ::tflite::micro::kTfLiteMmapRo;
using ::tflite::micro::kTfLiteMultiAxisQuantization;
using ::tflite::micro::kTfLiteNonCpu;
using ::tflite::micro::kTfLiteNoQuantization;
using ::tflite::micro::kTfLitePersistentRo;
using ::tflite::micro::kTfLiteVariantObject;
using ::tflite::micro::TfLiteAffineQuantization;
using ::tflite::micro::TfLiteAllocationType;
using ::tflite::micro::TfLiteBFloat16;
using ::tflite::micro::TfLiteBlockwiseQuantization;
using ::tflite::micro::TfLiteComplex128;
using ::tflite::micro::TfLiteComplex64;
using ::tflite::micro::TfLiteContext;
using ::tflite::micro::TfLiteEvalTensor;
using ::tflite::micro::TfLiteExternalContext;
using ::tflite::micro::TfLiteExternalContextType;
using ::tflite::micro::TfLiteFloat16;
using ::tflite::micro::TfLiteFloatArray;
using ::tflite::micro::TfLiteIntArray;
using ::tflite::micro::TfLiteIntArrayEqual;
using ::tflite::micro::TfLiteIntArrayEqualsArray;
using ::tflite::micro::TfLiteIntArrayGetSizeInBytes;
using ::tflite::micro::TfLiteMultiAxisQuantization;
using ::tflite::micro::TfLiteNode;
using ::tflite::micro::TfLitePtrUnion;
using ::tflite::micro::TfLiteQuantization;
using ::tflite::micro::TfLiteQuantizationType;
using ::tflite::micro::TfLiteTensor;
using ::tflite::micro::TfLiteTypeGetName;
#endif
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_MICRO_C_COMMON_H_
