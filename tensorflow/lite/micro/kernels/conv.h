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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_CONV_H_

#include <cstdint>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/op_dispatcher.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

struct OpDataConv {
  TfLitePaddingValues padding;

  // Cached tensor zero point values for quantized operations.
  int32_t input_zero_point;
  int32_t filter_zero_point;
  int32_t output_zero_point;

  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  // A buffer used to store unpacked filter values. This is used if the source
  // tensor is of n-bit precision that cannot be easily processed by kernels.
  int filter_buffer_index;

#ifdef USE_TFLM_COMPRESSION

  // scratch buffers for compressed tensors
  int weights_scratch_index;
  int bias_scratch_index;

#endif  // USE_TFLM_COMPRESSION
};

extern const int kConvInputTensor;
extern const int kConvWeightsTensor;
extern const int kConvBiasTensor;
extern const int kConvOutputTensor;
extern const int kConvQuantizedDimension;

// Returns a ConvParams struct with all the parameters needed for a
// float computation.
ConvParams ConvParamsFloat(const TfLiteConvParams& params,
                           const OpDataConv& data);

// Returns a ConvParams struct with all the parameters needed for a
// quantized computation.
ConvParams ConvParamsQuantized(const TfLiteConvParams& params,
                               const OpDataConv& data);

TfLiteStatus CalculateOpDataConv(TfLiteContext* context, TfLiteNode* node,
                                 const TfLiteConvParams& params, int width,
                                 int height, int filter_width,
                                 int filter_height, int out_width,
                                 int out_height, const TfLiteType data_type,
                                 OpDataConv* data);

void* ConvInit(TfLiteContext* context, const char* buffer, size_t length);

TfLiteStatus ConvPrepare(TfLiteContext* context, TfLiteNode* node);

// Backend contract for CONV_2D:
// Every CONV_2D backend implementation (reference, cmsis_nn, xtensa, arc_mli,
// ceva, etc.) must define these 6 entry points in `namespace tflite::micro`:
//   - ConvBackendInit: allocates backend-specific OpData
//   - ConvBackendPrepare: shared Prepare across all CONV_2D type variants
//   - ConvEvalFloat32: Eval for (input=float32, filter=float32)
//   - ConvEvalInt8: Eval for (input=int8, filter=int8)
//   - ConvEvalInt16x8: Eval for (input=int16, filter=int8)
//   - ConvEvalInt8x4: Eval for (input=int8, filter=int4)
// Splitting only the per-type Eval functions (while sharing Prepare) allows
// single-type registrations (`AddConv2D<conv::Int8>()`) to garbage-collect
// unused Eval implementations under `--function-sections` + `--gc-sections`.
void* ConvBackendInit(TfLiteContext* context, const char* buffer,
                      size_t length);
TfLiteStatus ConvBackendPrepare(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus ConvEvalFloat32(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus ConvEvalInt8(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus ConvEvalInt16x8(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus ConvEvalInt8x4(TfLiteContext* context, TfLiteNode* node);

// Type signatures for CONV_2D: (InputType, WeightType).
namespace conv {
using Float32 = OpSig<kTfLiteFloat32, kTfLiteFloat32>;
using Int8 = OpSig<kTfLiteInt8, kTfLiteInt8>;
using Int16x8 = OpSig<kTfLiteInt16, kTfLiteInt8>;
using Int8x4 = OpSig<kTfLiteInt8, kTfLiteInt4>;
}  // namespace conv

struct Conv2DOp {
  static constexpr size_t kKeyArity = 2;
  static constexpr auto Init = &ConvBackendInit;
  static constexpr auto Prepare = &ConvBackendPrepare;
  static bool Key(TfLiteContext* context, const TfLiteNode* node,
                  TfLiteType (&key)[2]);
};

template <>
struct OpVariant<Conv2DOp, conv::Float32> {
  static constexpr auto Eval = &ConvEvalFloat32;
};

template <>
struct OpVariant<Conv2DOp, conv::Int8> {
  static constexpr auto Eval = &ConvEvalInt8;
};

template <>
struct OpVariant<Conv2DOp, conv::Int16x8> {
  static constexpr auto Eval = &ConvEvalInt16x8;
};

template <>
struct OpVariant<Conv2DOp, conv::Int8x4> {
  static constexpr auto Eval = &ConvEvalInt8x4;
};

// This is the most generic TFLMRegistration. The actual supported types
// may still be target dependent. The only requirement is that every
// implementation (reference or optimized) must define this function.
TFLMRegistration Register_CONV_2D();

// Template overload for typed op registration of one or more OpSigs.
template <typename Sig1, typename... MoreSigs>
TFLMRegistration Register_CONV_2D() {
  return OpDispatcher<Conv2DOp, Sig1, MoreSigs...>::GetRegistration();
}

#if defined(XTENSA)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 activations and int8 weights and always calls the reference
// implementation.
TFLMRegistration Register_CONV_2D_INT8REF();

#else
inline TFLMRegistration Register_CONV_2D_INT8REF() {
  return Register_CONV_2D<conv::Int8>();
}
#endif  // defined(XTENSA)

#if defined(CMSIS_NN)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 activations and int4 weights and uses the latency optimized
// implementations.
TFLMRegistration Register_CONV_2D_INT4();
#else
inline TFLMRegistration Register_CONV_2D_INT4() {
  return Register_CONV_2D<conv::Int8x4>();
}
#endif  // defined(CMSIS_NN)

#if defined(CMSIS_NN) || defined(XTENSA)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 activations and int8 weights and uses the latency optimized
// implementations.
TFLMRegistration Register_CONV_2D_INT8();

// Returns a TFLMRegistration struct for kernel variant that only supports
// int16 activations and int8 weights and uses the latency optimized
// implementations.
TFLMRegistration Register_CONV_2D_INT16();

#else
inline TFLMRegistration Register_CONV_2D_INT8() {
  return Register_CONV_2D<conv::Int8>();
}

inline TFLMRegistration Register_CONV_2D_INT16() {
  return Register_CONV_2D<conv::Int16x8>();
}
#endif  // defined(CMSIS_NN) || defined(XTENSA)

}  // namespace micro

namespace conv = micro::conv;

using micro::CalculateOpDataConv;
using micro::Conv2DOp;
using micro::ConvBackendInit;
using micro::ConvBackendPrepare;
using micro::ConvEvalFloat32;
using micro::ConvEvalInt16x8;
using micro::ConvEvalInt8;
using micro::ConvEvalInt8x4;
using micro::ConvInit;
using micro::ConvParamsFloat;
using micro::ConvParamsQuantized;
using micro::ConvPrepare;
using micro::kConvBiasTensor;
using micro::kConvInputTensor;
using micro::kConvOutputTensor;
using micro::kConvQuantizedDimension;
using micro::kConvWeightsTensor;
using micro::OpDataConv;
using micro::Register_CONV_2D;
using micro::Register_CONV_2D_INT16;
using micro::Register_CONV_2D_INT4;
using micro::Register_CONV_2D_INT8;
using micro::Register_CONV_2D_INT8REF;

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_CONV_H_
