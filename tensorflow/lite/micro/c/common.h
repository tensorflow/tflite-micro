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

#if !defined(TENSORFLOW_LITE_CORE_C_COMMON_H_) && \
    !defined(TENSORFLOW_LITE_C_COMMON_H_)
#define TENSORFLOW_LITE_CORE_C_COMMON_H_
#define TENSORFLOW_LITE_C_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct TfLiteContext;
struct TfLiteDelegate;
struct TfLiteRegistration;
struct TfLiteNode;

#define kTfLiteOptionalTensor (-1)

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

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_KERNEL_LOG(context, ...)            \
  do {                                              \
    (context)->ReportError((context), __VA_ARGS__); \
  } while (false)

#define TF_LITE_MAYBE_KERNEL_LOG(context, ...)        \
  do {                                                \
    if ((context) != nullptr) {                       \
      (context)->ReportError((context), __VA_ARGS__); \
    }                                                 \
  } while (false)
#else
#define TF_LITE_KERNEL_LOG(context, ...)
#define TF_LITE_MAYBE_KERNEL_LOG(context, ...)
#endif

#define TF_LITE_ENSURE_STATUS(a) \
  do {                           \
    TfLiteStatus s = (a);        \
    if (s != kTfLiteOk) {        \
      return s;                  \
    }                            \
  } while (0)

#define TF_LITE_ENSURE(context, a)                                      \
  do {                                                                  \
    if (!(a)) {                                                         \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s was not true.", __FILE__, \
                         __LINE__, #a);                                 \
      return kTfLiteError;                                              \
    }                                                                   \
  } while (0)

#define TF_LITE_ENSURE_MSG(context, value, ...)                \
  do {                                                         \
    if (!(value)) {                                            \
      TF_LITE_KERNEL_LOG((context), __FILE__ " " __VA_ARGS__); \
      return kTfLiteError;                                     \
    }                                                          \
  } while (0)

#define TF_LITE_ENSURE_OK(context, a) \
  do {                                \
    TfLiteStatus s = (a);             \
    if (s != kTfLiteOk) {             \
      return s;                       \
    }                                 \
  } while (0)

#define TF_LITE_ENSURE_EQ(context, a, b)                                   \
  do {                                                                     \
    if ((a) != (b)) {                                                      \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s != %s (%d != %d)", __FILE__, \
                         __LINE__, #a, #b, (a), (b));                      \
      return kTfLiteError;                                                 \
    }                                                                      \
  } while (0)

#define TF_LITE_ENSURE_TYPES_EQ(context, a, b)                             \
  do {                                                                     \
    if ((a) != (b)) {                                                      \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s != %s (%s != %s)", __FILE__, \
                         __LINE__, #a, #b, TfLiteTypeGetName(a),           \
                         TfLiteTypeGetName(b));                            \
      return kTfLiteError;                                                 \
    }                                                                      \
  } while (0)

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

#ifdef __cplusplus
typedef enum TfLiteQuantizationType : int {
#else
typedef enum TfLiteQuantizationType {
#endif
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

typedef struct TfLiteDimensionMetadata {
  TfLiteDimensionType format;
  int dense_size;
  TfLiteIntArray* array_segments;
  TfLiteIntArray* array_indices;
} TfLiteDimensionMetadata;

typedef struct TfLiteSparsity {
  TfLiteIntArray* traversal_order;
  TfLiteIntArray* block_map;
  TfLiteDimensionMetadata* dim_metadata;
  int dim_metadata_size;
} TfLiteSparsity;

typedef struct TfLiteCustomAllocation {
  void* data;
  size_t bytes;
} TfLiteCustomAllocation;

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
  TfLiteStatus (*Refresh)(struct TfLiteContext* context);
} TfLiteExternalContext;

typedef int TfLiteBufferHandle;
enum {
  kTfLiteNullBufferHandle = -1,
};

#ifndef TF_LITE_STATIC_MEMORY
typedef struct TfLiteAllocator {
  void* data;
  void* (*allocate)(void* data, size_t bytes, size_t alignment);
  void* (*reallocate)(void* data, void* ptr, size_t old_bytes, size_t new_bytes,
                      size_t alignment);
  void (*deallocate)(void* data, void* ptr, size_t bytes, size_t alignment);
} TfLiteAllocator;

typedef enum TfLiteCustomAllocationFlags {
  kTfLiteCustomAllocationFlagsNone = 0,
  kTfLiteCustomAllocationFlagsSkipAlignCheck = 1,
} TfLiteCustomAllocationFlags;

enum { kTfLiteNoBufferIdentifier = SIZE_MAX };

typedef struct TfLiteTensor {
  TfLiteType type;
  TfLitePtrUnion data;
  TfLiteIntArray* dims;
  TfLiteQuantizationParams params;
  TfLiteAllocationType allocation_type;
  size_t bytes;
  const void* allocation;
  const char* name;
  struct TfLiteDelegate* delegate;
  TfLiteBufferHandle buffer_handle;
  bool data_is_stale;
  bool is_variable;
  TfLiteQuantization quantization;
  TfLiteSparsity* sparsity;
  const TfLiteIntArray* dims_signature;
} TfLiteTensor;

inline void TfLiteTensorDataFree(TfLiteTensor* t) {}
// Retained for LiteRT header compatibility in hybrid translation units that
// include this header before LiteRT headers (until shared header guards are
// removed).
void TfLiteTensorFree(TfLiteTensor* t);
TfLiteIntArray* TfLiteIntArrayCreate(int size);
void TfLiteIntArrayFree(TfLiteIntArray* a);
TfLiteFloatArray* TfLiteFloatArrayCreate(int size);
void TfLiteFloatArrayFree(TfLiteFloatArray* a);

typedef struct TfLiteEvalTensor {
  TfLitePtrUnion data;
  TfLiteIntArray* dims;
  TfLiteType type;
} TfLiteEvalTensor;

typedef struct TfLiteNode {
  TfLiteIntArray* inputs;
  TfLiteIntArray* outputs;
  TfLiteIntArray* intermediates;
  TfLiteIntArray* temporaries;
  void* user_data;
  void* builtin_data;
  const void* custom_initial_data;
  int custom_initial_data_size;
  struct TfLiteDelegate* delegate;
  bool might_have_side_effect;
} TfLiteNode;
#else   // defined(TF_LITE_STATIC_MEMORY)?
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

inline void TfLiteTensorDataFree(TfLiteTensor* t) {}

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
#endif  // TF_LITE_STATIC_MEMORY

typedef struct TfLiteOperator TfLiteOperator;
typedef TfLiteOperator TfLiteRegistrationExternal;

typedef struct TfLiteRegistration {
  void* (*init)(struct TfLiteContext* context, const char* buffer,
                size_t length);
  void (*free)(struct TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(struct TfLiteContext* context,
                          struct TfLiteNode* node);
  TfLiteStatus (*invoke)(struct TfLiteContext* context,
                         struct TfLiteNode* node);
  const char* (*profiling_string)(const struct TfLiteContext* context,
                                  const struct TfLiteNode* node);
  int32_t builtin_code;
  const char* custom_name;
  int version;
  TfLiteOperator* registration_external;
  struct TfLiteAsyncKernel* (*async_kernel)(struct TfLiteContext* context,
                                            struct TfLiteNode* node);
  uint64_t inplace_operator;
} TfLiteRegistration;

typedef struct TfLiteRegistration TfLiteRegistration_V1;

typedef struct TfLiteDelegateParams TfLiteDelegateParams;
typedef struct TfLiteDelegate TfLiteDelegate;

typedef struct TfLiteContext {
  size_t tensors_size;

  TfLiteStatus (*GetExecutionPlan)(struct TfLiteContext* context,
                                   TfLiteIntArray** execution_plan);

  struct TfLiteTensor* tensors;

  void* impl_;

  TfLiteStatus (*ResizeTensor)(struct TfLiteContext*,
                               struct TfLiteTensor* tensor,
                               TfLiteIntArray* new_size);
  void (*ReportError)(struct TfLiteContext*, const char* msg, ...);

  TfLiteStatus (*AddTensors)(struct TfLiteContext*, int tensors_to_add,
                             int* first_new_tensor_index);

  TfLiteStatus (*GetNodeAndRegistration)(
      struct TfLiteContext*, int node_index, struct TfLiteNode** node,
      struct TfLiteRegistration** registration);

  TfLiteStatus (*ReplaceNodeSubsetsWithDelegateKernels)(
      struct TfLiteContext*, struct TfLiteRegistration registration,
      const TfLiteIntArray* nodes_to_replace, struct TfLiteDelegate* delegate);

  int recommended_num_threads;

  TfLiteExternalContext* (*GetExternalContext)(struct TfLiteContext*,
                                               TfLiteExternalContextType);
  void (*SetExternalContext)(struct TfLiteContext*, TfLiteExternalContextType,
                             TfLiteExternalContext*);

  bool allow_fp32_relax_to_fp16;

  void* profiler;

  void* (*AllocatePersistentBuffer)(struct TfLiteContext* ctx, size_t bytes);

  TfLiteStatus (*AllocateBufferForEval)(struct TfLiteContext* ctx, size_t bytes,
                                        void** ptr);

  TfLiteStatus (*RequestScratchBufferInArena)(struct TfLiteContext* ctx,
                                              size_t bytes, int* buffer_idx);

  void* (*GetScratchBuffer)(struct TfLiteContext* ctx, int buffer_idx);

  TfLiteStatus (*ResizeTensorExplicit)(struct TfLiteContext* ctx,
                                       struct TfLiteTensor* tensor, int dims,
                                       const int* shape);

  TfLiteStatus (*PreviewDelegatePartitioning)(
      struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
      TfLiteDelegateParams** partition_params_array, int* num_partitions);

  struct TfLiteTensor* (*GetTensor)(const struct TfLiteContext* context,
                                    int tensor_idx);

  struct TfLiteEvalTensor* (*GetEvalTensor)(const struct TfLiteContext* context,
                                            int tensor_idx);

  TfLiteStatus (*GetModelMetadata)(const struct TfLiteContext* context,
                                   const char* name, const char** ptr,
                                   size_t* bytes);

  TfLiteStatus (*AcquireSubgraphContext)(
      struct TfLiteContext* context, int subgraph_index,
      struct TfLiteContext** acquired_context);
  TfLiteStatus (*ReleaseSubgraphContext)(struct TfLiteContext* context,
                                         int subgraph_index);
#if defined(_WIN32)
  TfLiteIntArray* (*TfLiteIntArrayCreate)(int size);  // NOLINT

  void (*TfLiteIntArrayFree)(TfLiteIntArray* a);  // NOLINT
#endif                                            // defined(_WIN32)
} TfLiteContext;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // !defined(TENSORFLOW_LITE_CORE_C_COMMON_H_)

#endif  // TENSORFLOW_LITE_MICRO_C_COMMON_H_
