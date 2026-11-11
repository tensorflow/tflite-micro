/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_SOFTMAX_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_SOFTMAX_H_

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/op_dispatcher.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {
void* SoftmaxInit(TfLiteContext* context, const char* buffer, size_t length);

// Common helper function to SoftmaxPrepare.
TfLiteStatus CalculateSoftmaxParams(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    TfLiteTensor* output,
                                    const TfLiteSoftmaxParams* params,
                                    SoftmaxParams* op_data);

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node);

// Backend contract for SOFTMAX:
// Every SOFTMAX backend implementation (reference, cmsis_nn, xtensa, ceva)
// must define these 6 entry points in `namespace tflite::micro`:
//   - SoftmaxBackendInit: allocates backend-specific OpData
//   - SoftmaxBackendPrepare: shared Prepare across all SOFTMAX type variants
//   - SoftmaxEvalFloat32: Eval for (input=float32, output=float32)
//   - SoftmaxEvalInt8: Eval for (input=int8, output=int8)
//   - SoftmaxEvalInt8To16: Eval for (input=int8, output=int16)
//   - SoftmaxEvalInt16: Eval for (input=int16, output=int16)
void* SoftmaxBackendInit(TfLiteContext* context, const char* buffer,
                         size_t length);
TfLiteStatus SoftmaxBackendPrepare(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus SoftmaxEvalFloat32(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus SoftmaxEvalInt8(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus SoftmaxEvalInt8To16(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus SoftmaxEvalInt16(TfLiteContext* context, TfLiteNode* node);

// Type signatures for SOFTMAX: (InputType, OutputType).
namespace softmax {
using Float32 = OpSig<kTfLiteFloat32, kTfLiteFloat32>;
using Int8 = OpSig<kTfLiteInt8, kTfLiteInt8>;
using Int8To16 = OpSig<kTfLiteInt8, kTfLiteInt16>;
using Int16 = OpSig<kTfLiteInt16, kTfLiteInt16>;
}  // namespace softmax

struct SoftmaxOp {
  static constexpr size_t kKeyArity = 2;
  static constexpr auto Init = &SoftmaxBackendInit;
  static constexpr auto Prepare = &SoftmaxBackendPrepare;
  static bool Key(TfLiteContext* context, const TfLiteNode* node,
                  TfLiteType (&key)[2]);
};

template <>
struct OpVariant<SoftmaxOp, softmax::Float32> {
  static constexpr auto Eval = &SoftmaxEvalFloat32;
};

template <>
struct OpVariant<SoftmaxOp, softmax::Int8> {
  static constexpr auto Eval = &SoftmaxEvalInt8;
};

template <>
struct OpVariant<SoftmaxOp, softmax::Int8To16> {
  static constexpr auto Eval = &SoftmaxEvalInt8To16;
};

template <>
struct OpVariant<SoftmaxOp, softmax::Int16> {
  static constexpr auto Eval = &SoftmaxEvalInt16;
};

// This is the most generic TFLMRegistration. The actual supported types
// may still be target dependent. The only requirement is that every
// implementation (reference or optimized) must define this function.
TFLMRegistration Register_SOFTMAX();

// Template overload for typed op registration of one or more OpSigs.
template <typename Sig1, typename... MoreSigs>
TFLMRegistration Register_SOFTMAX() {
  return OpDispatcher<SoftmaxOp, Sig1, MoreSigs...>::GetRegistration();
}

#if defined(XTENSA) || defined(CMSIS_NN)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 input and int16 output.
TFLMRegistration Register_SOFTMAX_INT8_INT16();
#else
inline TFLMRegistration Register_SOFTMAX_INT8_INT16() {
  return Register_SOFTMAX<softmax::Int8To16>();
}
#endif

#if defined(CMSIS_NN)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 input/output and uses the latency optimized implementations.
TFLMRegistration Register_SOFTMAX_INT8();

// Returns a TFLMRegistration struct for kernel variant that only supports
// int16 input/output and uses the latency optimized implementations.
TFLMRegistration Register_SOFTMAX_INT16();

#else
inline TFLMRegistration Register_SOFTMAX_INT8() {
  return Register_SOFTMAX<softmax::Int8>();
}

inline TFLMRegistration Register_SOFTMAX_INT16() {
  return Register_SOFTMAX<softmax::Int16>();
}
#endif

}  // namespace micro

namespace softmax = micro::softmax;

using micro::Register_SOFTMAX;
using micro::Register_SOFTMAX_INT16;
using micro::Register_SOFTMAX_INT8;
using micro::Register_SOFTMAX_INT8_INT16;
using micro::SoftmaxBackendInit;
using micro::SoftmaxBackendPrepare;
using micro::SoftmaxEvalFloat32;
using micro::SoftmaxEvalInt16;
using micro::SoftmaxEvalInt8;
using micro::SoftmaxEvalInt8To16;
using micro::SoftmaxOp;
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_SOFTMAX_H_
