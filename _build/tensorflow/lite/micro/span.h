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
*/

#ifndef TENSORFLOW_LITE_MICRO_SPAN_H_
#define TENSORFLOW_LITE_MICRO_SPAN_H_

#include <cstddef>

namespace tflite {

// A poor man's std::span, we should consider using the Pigweed span instead.
template <typename T>
class Span {
 public:
  constexpr Span(T* data, size_t size) noexcept : data_(data), size_(size) {}

  constexpr T& operator[](size_t idx) const noexcept { return *(data_ + idx); }

  constexpr T* data() const noexcept { return data_; }
  constexpr size_t size() const noexcept { return size_; }

 private:
  T* data_;
  size_t size_;
};

}  // end namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_SPAN_H_
