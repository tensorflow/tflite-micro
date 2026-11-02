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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_OP_DISPATCHER_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_OP_DISPATCHER_H_

#include <cstddef>
#include <type_traits>
#include <utility>

#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {

// Compile-time tensor type signature for an operator variant.
template <TfLiteType... Ts>
struct OpSig {
  static_assert(sizeof...(Ts) > 0, "OpSig requires at least one TfLiteType.");
  static constexpr size_t kArity = sizeof...(Ts);
  static constexpr TfLiteType kTypes[sizeof...(Ts)] = {Ts...};
};

// Primary trait mapping (OpTag, Sig) -> {Eval} function pointer.
// Note: All OpVariant<OpTag, Sig> specializations for an operator must be
// declared in that operator's header before OpDispatcher<OpTag, ...> is
// instantiated.
template <typename OpTag, typename Sig>
struct OpVariant;

// Null-safe, bounds-checked eval tensor type helpers that use
// ctx->GetEvalTensor directly without pulling in kernel_util.h.
inline bool GetEvalInputType(TfLiteContext* ctx, const TfLiteNode* node,
                             int index, TfLiteType* out_type) {
  if (ctx == nullptr || ctx->GetEvalTensor == nullptr || node == nullptr ||
      node->inputs == nullptr || index < 0 || index >= node->inputs->size) {
    return false;
  }
  const int tensor_index = node->inputs->data[index];
  if (tensor_index == kTfLiteOptionalTensor) {
    return false;
  }
  const TfLiteEvalTensor* tensor = ctx->GetEvalTensor(ctx, tensor_index);
  if (tensor == nullptr) {
    return false;
  }
  *out_type = tensor->type;
  return true;
}

inline bool GetEvalOutputType(TfLiteContext* ctx, const TfLiteNode* node,
                              int index, TfLiteType* out_type) {
  if (ctx == nullptr || ctx->GetEvalTensor == nullptr || node == nullptr ||
      node->outputs == nullptr || index < 0 || index >= node->outputs->size) {
    return false;
  }
  const int tensor_index = node->outputs->data[index];
  if (tensor_index == kTfLiteOptionalTensor) {
    return false;
  }
  const TfLiteEvalTensor* tensor = ctx->GetEvalTensor(ctx, tensor_index);
  if (tensor == nullptr) {
    return false;
  }
  *out_type = tensor->type;
  return true;
}

namespace detail {

template <typename OpTag, typename Sig, typename = void>
struct HasVariantImpl : std::false_type {};

template <typename OpTag, typename Sig>
struct HasVariantImpl<OpTag, Sig,
                      std::void_t<decltype(OpVariant<OpTag, Sig>::Eval)>>
    : std::true_type {};

template <typename T, typename... Rest>
constexpr bool HasDuplicates() {
  if constexpr (sizeof...(Rest) == 0) {
    return false;
  } else {
    return (std::is_same_v<T, Rest> || ...) || HasDuplicates<Rest...>();
  }
}

template <typename Sig, size_t... Is>
constexpr bool MatchKeyImpl(const TfLiteType (&key)[sizeof...(Is)],
                            std::index_sequence<Is...>) {
  static_assert(Sig::kArity == sizeof...(Is),
                "OpSig arity does not match OpTag::Key size.");
  return ((Sig::kTypes[Is] == kTfLiteNoType || key[Is] == Sig::kTypes[Is]) &&
          ...);
}

template <typename Sig, size_t N>
constexpr bool MatchKey(const TfLiteType (&key)[N]) {
  return MatchKeyImpl<Sig>(key, std::make_index_sequence<N>{});
}

template <typename OpTag, typename = void>
struct HasFree : std::false_type {};

template <typename OpTag>
struct HasFree<OpTag, std::void_t<decltype(OpTag::Free)>> : std::true_type {};

template <typename OpTag, typename = void>
struct HasReset : std::false_type {};

template <typename OpTag>
struct HasReset<OpTag, std::void_t<decltype(OpTag::Reset)>> : std::true_type {};

}  // namespace detail

// Checks whether OpVariant<OpTag, Sig> is specialized with an Eval member.
// All OpVariant<OpTag, Sig> specializations for an op must be declared in the
// op's header prior to instantiating HasVariant or OpDispatcher.
template <typename OpTag, typename Sig>
inline constexpr bool HasVariant = detail::HasVariantImpl<OpTag, Sig>::value;

template <typename OpTag, typename... Sigs>
struct OpDispatcher {
  static_assert(sizeof...(Sigs) > 0, "At least one OpSig is required.");
  static_assert((HasVariant<OpTag, Sigs> && ...),
                "Requested OpSig is not supported by this operator.");
  static_assert(!detail::HasDuplicates<Sigs...>(),
                "Duplicate OpSig in OpDispatcher.");

  static constexpr size_t kKeyArity = OpTag::kKeyArity;

  static TfLiteStatus Prepare(TfLiteContext* ctx, TfLiteNode* node) {
    TfLiteType key[kKeyArity] = {};
    if (!OpTag::Key(ctx, node, key)) {
      MicroPrintf("Failed to extract valid tensor type key in Prepare.");
      return kTfLiteError;
    }
    const bool matched = (detail::MatchKey<Sigs>(key) || ...);
    if (!matched) {
      if constexpr (kKeyArity >= 2) {
        MicroPrintf(
            "Unsupported tensor type combination (%s, %s) for registered op.",
            TfLiteTypeGetName(key[0]), TfLiteTypeGetName(key[1]));
      } else {
        MicroPrintf("Unsupported tensor type (%s) for registered op.",
                    TfLiteTypeGetName(key[0]));
      }
      return kTfLiteError;
    }
    return OpTag::Prepare(ctx, node);
  }

  static TfLiteStatus Eval(TfLiteContext* ctx, TfLiteNode* node) {
    TfLiteType key[kKeyArity] = {};
    if (!OpTag::Key(ctx, node, key)) {
      return kTfLiteError;
    }
    TfLiteStatus status = kTfLiteError;
    const bool matched =
        ((detail::MatchKey<Sigs>(key)
              ? (status = OpVariant<OpTag, Sigs>::Eval(ctx, node), true)
              : false) ||
         ...);
    return matched ? status : kTfLiteError;
  }

  static constexpr auto SelectFree() {
    if constexpr (detail::HasFree<OpTag>::value) {
      return OpTag::Free;
    } else {
      return nullptr;
    }
  }

  static constexpr auto SelectReset() {
    if constexpr (detail::HasReset<OpTag>::value) {
      return OpTag::Reset;
    } else {
      return nullptr;
    }
  }

  static TFLMRegistration GetRegistration() {
    TFLMRegistration r = {};
    r.init = OpTag::Init;
    r.free = SelectFree();
    r.prepare = &Prepare;
    if constexpr (sizeof...(Sigs) == 1) {
      // Single-type fast path: Prepare() validates the tensor type key once,
      // and invoke points directly to the variant's Eval function without any
      // dispatcher Eval wrapper.
      r.invoke = OpVariant<OpTag, Sigs...>::Eval;
    } else {
      r.invoke = &Eval;
    }
    r.reset = SelectReset();
    return r;
  }
};

}  // namespace micro

using micro::OpDispatcher;
using micro::OpSig;
using micro::OpVariant;

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_OP_DISPATCHER_H_
