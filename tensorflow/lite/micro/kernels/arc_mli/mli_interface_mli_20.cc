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

#include <math.h>

#include "mli_interface.h"  // NOLINT
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace ops {
namespace micro {

#ifdef MLI_2_0
template <>
int8_t* MliTensorInterface::Data(void) {
  TFLITE_DCHECK(tensor_->el_type == MLI_EL_SA_8);
  return tensor_->data.mem.pi8;
}

template <>
int32_t* MliTensorInterface::Data(void) {
  TFLITE_DCHECK(tensor_->el_type == MLI_EL_SA_32);
  return tensor_->data.mem.pi32;
}

template <>
int16_t** MliTensorInterface::Scale(void) {
  return &tensor_->el_params.sa.scale.mem.pi16;
}

template <>
int16_t* MliTensorInterface::Scale(void) {
  return &tensor_->el_params.sa.scale.mem.i16;
}

template <>
void MliTensorInterface::SetData(int8_t* data, uint32_t capacity) const {
  TFLITE_DCHECK(tensor_->el_type == MLI_EL_SA_8);
  tensor_->data.mem.pi8 = data;
  tensor_->data.capacity = capacity;
}

template <>
void MliTensorInterface::SetData(int32_t* data, uint32_t capacity) const {
  TFLITE_DCHECK(tensor_->el_type == MLI_EL_SA_32);
  tensor_->data.mem.pi32 = data;
  tensor_->data.capacity = capacity;
}

mli_tensor* MliTensorInterface::MliTensor(void) { return tensor_; }

const mli_tensor* MliTensorInterface::MliTensor(void) const {
  return static_cast<const mli_tensor*>(
      const_cast<MliTensorInterface*>(this)->MliTensor());
}

uint32_t* MliTensorInterface::Rank(void) { return &tensor_->rank; }

const uint32_t* MliTensorInterface::DataCapacity(void) const {
  return &tensor_->data.capacity;
}

mli_element_type* MliTensorInterface::ElType(void) { return &tensor_->el_type; }

template <>
int16_t* MliTensorInterface::ZeroPoint(void) {
  return &tensor_->el_params.sa.zero_point.mem.i16;
}

template <>
int16_t** MliTensorInterface::ZeroPoint(void) {
  return &tensor_->el_params.sa.zero_point.mem.pi16;
}

uint32_t* MliTensorInterface::ZeroPointCapacity(void) {
  return &tensor_->el_params.sa.zero_point.capacity;
}

int32_t* MliTensorInterface::Dim(void) { return &tensor_->el_params.sa.dim; }

uint32_t* MliTensorInterface::ScaleCapacity(void) {
  return &tensor_->el_params.sa.scale.capacity;
}

template <>
int8_t** MliTensorInterface::ScaleFracBits(void) {
  return &tensor_->el_params.sa.scale_frac_bits.mem.pi8;
}

template <>
int8_t* MliTensorInterface::ScaleFracBits(void) {
  return &tensor_->el_params.sa.scale_frac_bits.mem.i8;
}

uint32_t* MliTensorInterface::ScaleFracBitsCapacity(void) {
  return &tensor_->el_params.sa.scale_frac_bits.capacity;
}

int32_t* MliTensorInterface::MemStride(void) { return tensor_->mem_stride; }

uint32_t* MliTensorInterface::Shape(void) { return tensor_->shape; }

const uint32_t* MliTensorInterface::Shape(void) const {
  return static_cast<const uint32_t*>(
      const_cast<MliTensorInterface*>(this)->Shape());
}

void MliTensorInterface::SetScale(float fscale) {
  int exp;
  frexpf(fscale, &exp);
  int frac_bits = 15 - exp;
  int16_t iscale = (int16_t)((1ll << frac_bits) * fscale + 0.5f);
  *(this->Scale<int16_t*>()) = (int16_t)iscale;
  *(this->ScaleFracBits<int8_t*>()) = frac_bits;
  *this->ScaleCapacity() = 0;
  *this->ScaleFracBitsCapacity() = 0;
}

void MliTensorInterface::SetScalePerChannel(float* fscale,
                                            const int num_channels) {
  for (int i = 0; i < num_channels; i++) {
    int exp;
    frexpf(fscale[i], &exp);
    int cur_frac_bits = 15 - exp;
    (*this->ScaleFracBits<int8_t**>())[i] = cur_frac_bits;
  }

  for (int i = 0; i < num_channels; i++) {
    int16_t iscale =
        (int16_t)((1ll << (*this->ScaleFracBits<int8_t**>())[i]) * fscale[i] +
                  0.5f);
    (*this->Scale<int16_t**>())[i] = iscale;
  }
  *this->ScaleCapacity() = num_channels * sizeof(int16_t);
  *this->ScaleFracBitsCapacity() = num_channels * sizeof(int8_t);
}

void MliTensorInterface::SetElType(TfLiteType type) {
  if (type == kTfLiteInt8) {
    *this->ElType() = MLI_EL_SA_8;
  } else if (type == kTfLiteInt32) {
    *this->ElType() = MLI_EL_SA_32;
  } else {
    MicroPrintf("Wrong data type. Expected int8_t or int32_t.");
    TFLITE_ABORT;
  }
}
#endif

}  // namespace micro
}  // namespace ops
}  // namespace tflite
