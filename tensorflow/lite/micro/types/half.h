/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TYPES_HALF_H_
#define TENSORFLOW_LITE_TYPES_HALF_H_

#include <cstdint>

namespace tflite {

class half {
 private:
  // We need this hoop jumping to enable implementing a constexpr `from_bits`.
  struct zero_initializer {};
  explicit constexpr half(zero_initializer) : bits_(0) {}

 public:
  half() = default;

  // Disabled in TFLM to avoid dependencies on external fp16 conversion
  // libraries. This is safe because TFLM does not currently support
  // Float16 kernels, meaning reference kernel templates are never instantiated
  // with `tflite::half`.
  //
  // If Float16 support is needed in the future, the build target must be
  // updated to depend on a proper fp16 library and this code re-enabled.
#if 0
  half(float x) : bits_(fp16_ieee_from_fp32_value(x)) {}  // NOLINT
  explicit half(int x)
      : bits_(fp16_ieee_from_fp32_value(static_cast<float>(x))) {}

  operator float() const { return fp16_ieee_to_fp32_value(bits_); }  // NOLINT
#endif

  static constexpr half from_bits(uint16_t bits) {
    half result{zero_initializer{}};
    result.bits_ = bits;
    return result;
  }

  constexpr uint16_t to_bits() const { return bits_; }

  bool is_zero() const {
    // Check for +/- zero (0x0000/0x8000). uint16 overflow is well defined to
    // wrap around.
    return static_cast<uint16_t>(bits_ * 2) == 0;
  }

  static constexpr half epsilon() {
    return half::from_bits(0x1400);  // 2^-10 = 0.0009765625
  }
  static constexpr half infinity() { return from_bits(0x7c00); }
  static constexpr half min() { return from_bits(0xfbff); }
  static constexpr half max() { return from_bits(0x7bff); }
  static constexpr half smallest_normal() {
    return from_bits(0x0400);  // 2^-14
  }
  static constexpr half min_identity() { return from_bits(0x7c00); }
  static constexpr half max_identity() { return from_bits(0xfc00); }
  static constexpr half sum_identity() { return from_bits(0); }

  // Not private due to -Werror=class-memaccess, which can't be disabled:
  // - via a --copt, because it seems to have no effect.
  // - via .bazelrc, because it then applies to C code, and the compiler says
  //   this flag is not valid in C.
  uint16_t bits_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_TYPES_HALF_H_
