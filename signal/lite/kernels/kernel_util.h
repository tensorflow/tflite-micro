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

#ifndef SIGNAL_LITE_KERNELS_KERNEL_UTIL_H_
#define SIGNAL_LITE_KERNELS_KERNEL_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace tflm_signal {
namespace lite {

// FlatBuffers TensorType values stored in custom op FlexBuffers "T" attributes.
constexpr int32_t kTensorTypeFloat32 = 0;
constexpr int32_t kTensorTypeInt32 = 2;
constexpr int32_t kTensorTypeInt16 = 7;

template <typename T>
inline T* GetTensorData(TfLiteOpaqueTensor* tensor) {
  return reinterpret_cast<T*>(TfLiteOpaqueTensorData(tensor));
}

template <typename T>
inline const T* GetTensorData(const TfLiteOpaqueTensor* tensor) {
  return reinterpret_cast<const T*>(TfLiteOpaqueTensorData(tensor));
}

int GetFlatSize(const TfLiteOpaqueTensor* tensor);

::TfLiteRegistration MakeRegistration(const char* custom_name,
                                      TfLiteOperator* op);

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_LITE_KERNELS_KERNEL_UTIL_H_
