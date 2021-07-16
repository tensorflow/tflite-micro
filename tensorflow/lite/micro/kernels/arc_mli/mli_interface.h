/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_INTERFACE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_INTERFACE_H_

#include "mli_api.h"  // NOLINT
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
namespace tflite {
namespace ops {
namespace micro {

// Abstracts access to mli_tensor fields to use different versions of MLI
// Library (1.x and 2.x)
// Example:
//    ops::micro::MliTensorInterface mli_in =
//    ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
//        context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));

class MliTensorInterface {
 public:
  // Make sure that lifetime of MliTensorInterface instance isn't bigger than
  // related mli_tensor.
  MliTensorInterface(mli_tensor* tensor) : tensor_(tensor){};
  MliTensorInterface() = default;
  ~MliTensorInterface() = default;

  template <typename T>
  T* Data();
  template <typename T>
  T Scale();
  template <typename T>
  T ZeroPoint();
  template <typename T>
  T ScaleFracBits();
  mli_tensor* MliTensor();
  const mli_tensor* MliTensor() const;
  int32_t* Dim();
  uint32_t* Rank();
  uint32_t* Shape();
  const uint32_t* Shape() const;
  const uint32_t* DataCapacity() const;
  uint32_t* ScaleCapacity();
  mli_element_type* ElType();
  uint32_t* ScaleFracBitsCapacity();
  int32_t* MemStride();
  uint32_t* ZeroPointCapacity();

  template <typename T>
  void SetData(T* data, uint32_t capacity) const;
  void SetScale(float fscale);
  void SetScalePerChannel(float* fscale, const int num_channels);
  void SetElType(TfLiteType type);

 private:
  mli_tensor* tensor_;
};

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_SLICERS_H_
