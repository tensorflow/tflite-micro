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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_COMMON_H_
#define TENSORFLOW_LITE_MICRO_MICRO_COMMON_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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
    if (delta > (epsilon)) {                                                 \
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
  kTfLiteAbort = 15,
} TfLiteStatus;

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
  kTfLitePersistentRo,
} TfLiteAllocationType;

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

#ifdef __cplusplus
void* MicroContextAllocatePersistentBuffer(TfLiteContext* ctx, size_t bytes);
TfLiteStatus MicroContextRequestScratchBufferInArena(TfLiteContext* ctx,
                                                     size_t bytes,
                                                     int* buffer_idx);
void* MicroContextGetScratchBuffer(TfLiteContext* ctx, int buffer_idx);
TfLiteTensor* MicroContextGetTensor(const TfLiteContext* context,
                                    int tensor_idx);
TfLiteEvalTensor* MicroContextGetEvalTensor(const TfLiteContext* context,
                                            int tensor_idx);
void MicroContextReportOpError(TfLiteContext* context, const char* format, ...);
#endif  // __cplusplus

typedef struct TfLiteContext {
#ifdef __cplusplus
  void* impl_ = nullptr;

  template <auto Fn>
  struct Method {
    using FnType = decltype(Fn);
    template <typename... Args>
    auto operator()(Args... args) const {
      return Fn(args...);
    }
    constexpr operator FnType() const noexcept { return Fn; }
    constexpr bool operator!=(decltype(nullptr)) const noexcept { return true; }
    constexpr bool operator==(decltype(nullptr)) const noexcept {
      return false;
    }
    friend constexpr bool operator!=(decltype(nullptr), Method) noexcept {
      return true;
    }
    friend constexpr bool operator==(decltype(nullptr), Method) noexcept {
      return false;
    }
    explicit constexpr operator bool() const noexcept { return true; }
  };

  static constexpr Method<&MicroContextReportOpError> ReportError{};
  static constexpr Method<&MicroContextAllocatePersistentBuffer>
      AllocatePersistentBuffer{};
  static constexpr Method<&MicroContextRequestScratchBufferInArena>
      RequestScratchBufferInArena{};
  static constexpr Method<&MicroContextGetScratchBuffer> GetScratchBuffer{};
  static constexpr Method<&MicroContextGetTensor> GetTensor{};
  static constexpr Method<&MicroContextGetEvalTensor> GetEvalTensor{};
#else
  void* impl_;
#endif  // __cplusplus
} TfLiteContext;

#ifdef __cplusplus
}  // namespace micro

using micro::kTfLiteAbort;
using micro::kTfLiteAffineQuantization;
using micro::kTfLiteArenaRw;
using micro::kTfLiteArenaRwPersistent;
using micro::kTfLiteBFloat16;
using micro::kTfLiteBlockwiseQuantization;
using micro::kTfLiteBool;
using micro::kTfLiteComplex128;
using micro::kTfLiteComplex64;
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
using micro::kTfLiteMemNone;
using micro::kTfLiteMmapRo;
using micro::kTfLiteMultiAxisQuantization;
using micro::kTfLiteNoQuantization;
using micro::kTfLiteNoType;
using micro::kTfLiteOk;
using micro::kTfLitePersistentRo;
using micro::kTfLiteResource;
using micro::kTfLiteString;
using micro::kTfLiteUInt16;
using micro::kTfLiteUInt32;
using micro::kTfLiteUInt4;
using micro::kTfLiteUInt64;
using micro::kTfLiteUInt8;
using micro::kTfLiteVariant;
using micro::MicroContextAllocatePersistentBuffer;
using micro::MicroContextGetEvalTensor;
using micro::MicroContextGetScratchBuffer;
using micro::MicroContextGetTensor;
using micro::MicroContextReportOpError;
using micro::MicroContextRequestScratchBufferInArena;
using micro::TfLiteAffineQuantization;
using micro::TfLiteAllocationType;
using micro::TfLiteBFloat16;
using micro::TfLiteBlockwiseQuantization;
using micro::TfLiteComplex128;
using micro::TfLiteComplex64;
using micro::TfLiteContext;
using micro::TfLiteEvalTensor;
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
using micro::TfLiteQuantizationParams;
using micro::TfLiteQuantizationType;
using micro::TfLiteStatus;
using micro::TfLiteTensor;
using micro::TfLiteType;
using micro::TfLiteTypeGetName;
using micro::TFLMInferenceRegistration;
using micro::TFLMRegistration;

}  // namespace tflite

using ::tflite::micro::TFLMInferenceRegistration;
using ::tflite::micro::TFLMRegistration;

#if !defined(TFLM_NO_GLOBAL_C_ALIASES) &&              \
    !defined(TENSORFLOW_LITE_CORE_C_C_API_TYPES_H_) && \
    !defined(TENSORFLOW_LITE_C_C_API_TYPES_H_)
using ::tflite::micro::kTfLiteAbort;
using ::tflite::micro::kTfLiteBFloat16;
using ::tflite::micro::kTfLiteBool;
using ::tflite::micro::kTfLiteComplex128;
using ::tflite::micro::kTfLiteComplex64;
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
using ::tflite::micro::kTfLiteResource;
using ::tflite::micro::kTfLiteString;
using ::tflite::micro::kTfLiteUInt16;
using ::tflite::micro::kTfLiteUInt32;
using ::tflite::micro::kTfLiteUInt4;
using ::tflite::micro::kTfLiteUInt64;
using ::tflite::micro::kTfLiteUInt8;
using ::tflite::micro::kTfLiteVariant;
using ::tflite::micro::TfLiteQuantizationParams;
using ::tflite::micro::TfLiteStatus;
using ::tflite::micro::TfLiteType;
#endif

#if !defined(TFLM_NO_GLOBAL_C_ALIASES) &&         \
    !defined(TENSORFLOW_LITE_CORE_C_COMMON_H_) && \
    !defined(TENSORFLOW_LITE_C_COMMON_H_)
using ::tflite::micro::kTfLiteAffineQuantization;
using ::tflite::micro::kTfLiteArenaRw;
using ::tflite::micro::kTfLiteArenaRwPersistent;
using ::tflite::micro::kTfLiteBlockwiseQuantization;
using ::tflite::micro::kTfLiteMemNone;
using ::tflite::micro::kTfLiteMmapRo;
using ::tflite::micro::kTfLiteMultiAxisQuantization;
using ::tflite::micro::kTfLiteNoQuantization;
using ::tflite::micro::kTfLitePersistentRo;
using ::tflite::micro::TfLiteAffineQuantization;
using ::tflite::micro::TfLiteAllocationType;
using ::tflite::micro::TfLiteBFloat16;
using ::tflite::micro::TfLiteBlockwiseQuantization;
using ::tflite::micro::TfLiteComplex128;
using ::tflite::micro::TfLiteComplex64;
using ::tflite::micro::TfLiteContext;
using ::tflite::micro::TfLiteEvalTensor;
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

#endif  // TENSORFLOW_LITE_MICRO_MICRO_COMMON_H_
