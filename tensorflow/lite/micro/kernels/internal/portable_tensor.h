/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_INTERNAL_PORTABLE_TENSOR_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_INTERNAL_PORTABLE_TENSOR_H_

#include <cstddef>
#include <cstring>

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"

namespace tflite {
namespace micro {

// Writes randomly accessed values from `input` sequentially into `output`.
template <typename T>
class SequentialTensorWriter {
 public:
  SequentialTensorWriter(const TfLiteTensor* input, TfLiteTensor* output) {
    input_data_ = GetTensorData<T>(input);
    output_ptr_ = GetTensorData<T>(output);
  }
  SequentialTensorWriter(const T* input_data, T* output_data)
      : input_data_(input_data), output_ptr_(output_data) {}

  void Write(int position) { *output_ptr_++ = input_data_[position]; }
  void WriteN(int position, int len) {
    memcpy(output_ptr_, &input_data_[position], sizeof(T) * len);
    output_ptr_ += len;
  }

 private:
  const T* input_data_;
  T* output_ptr_;
};

}  // namespace micro

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_H_
using micro::SequentialTensorWriter;
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_H_

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_INTERNAL_PORTABLE_TENSOR_H_
