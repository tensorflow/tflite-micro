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

#include "mli_interface.h"  // NOLINT

#include <math.h>

#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace ops {
namespace micro {

#ifndef MLI_2_0
template <>
int8_t* MliTensorInterface::Data<int8_t>(void) {
  TFLITE_DCHECK(tensor_->el_type == MLI_EL_ASYM_I8);
  return static_cast<int8_t*>(tensor_->data);
}

template <>
int32_t* MliTensorInterface::Data<int32_t>(void) {
  TFLITE_DCHECK(tensor_->el_type == MLI_EL_ASYM_I32);
  return static_cast<int32_t*>(tensor_->data);
}

template <>
int32_t* MliTensorInterface::Scale(void) {
  return &tensor_->el_params.asym.scale.i32;
}

template <>
int32_t** MliTensorInterface::Scale(void) {
  return &tensor_->el_params.asym.scale.pi32;
}

template <>
void MliTensorInterface::SetData(int8_t* data, uint32_t capacity) const {
  TFLITE_DCHECK(tensor_->el_type == MLI_EL_ASYM_I8);
  tensor_->data = data;
  tensor_->capacity = capacity;
}

template <>
void MliTensorInterface::SetData(int32_t* data, uint32_t capacity) const {
  TFLITE_DCHECK(tensor_->el_type == MLI_EL_ASYM_I32);
  tensor_->data = data;
  tensor_->capacity = capacity;
}

mli_tensor* MliTensorInterface::MliTensor(void) { return tensor_; }

const mli_tensor* MliTensorInterface::MliTensor(void) const {
  return static_cast<const mli_tensor*>(
      const_cast<MliTensorInterface*>(this)->MliTensor());
}

uint32_t* MliTensorInterface::Rank(void) { return &tensor_->rank; }

const uint32_t* MliTensorInterface::DataCapacity(void) const {
  return &tensor_->capacity;
}

mli_element_type* MliTensorInterface::ElType(void) { return &tensor_->el_type; }

template <>
int16_t* MliTensorInterface::ZeroPoint(void) {
  return &tensor_->el_params.asym.zero_point.i16;
}

template <>
int16_t** MliTensorInterface::ZeroPoint(void) {
  return &tensor_->el_params.asym.zero_point.pi16;
}

uint32_t* MliTensorInterface::ZeroPointCapacity(void) { return nullptr; }

int32_t* MliTensorInterface::Dim(void) { return &tensor_->el_params.asym.dim; }

uint32_t* MliTensorInterface::ScaleCapacity(void) { return nullptr; }

template <>
int8_t* MliTensorInterface::ScaleFracBits(void) {
  return &tensor_->el_params.asym.scale_frac_bits;
}

uint32_t* MliTensorInterface::ScaleFracBitsCapacity(void) { return nullptr; }

int32_t* MliTensorInterface::MemStride(void) { return tensor_->mem_stride; }

uint32_t* MliTensorInterface::Shape(void) { return tensor_->shape; }

const uint32_t* MliTensorInterface::Shape(void) const {
  return static_cast<const uint32_t*>(
      const_cast<MliTensorInterface*>(this)->Shape());
}

void MliTensorInterface::SetScale(float fscale) {
  int exp;
  frexpf(fscale, &exp);
  int frac_bits = 31 - exp;
  int32_t iscale = (int32_t)((1ll << frac_bits) * fscale + 0.5f);
  *(this->ScaleFracBits<int8_t*>()) = frac_bits;
  *(this->Scale<int32_t*>()) = (int32_t)iscale;
}

void MliTensorInterface::SetScalePerChannel(float* fscale,
                                            const int num_channels) {
  int min_frac_bits;
  for (int i = 0; i < num_channels; i++) {
    int exp;
    frexpf(fscale[i], &exp);
    int cur_frac_bits = 31 - exp;
    if (i == 0) {
      min_frac_bits = cur_frac_bits;
    } else {
      min_frac_bits =
          min_frac_bits < cur_frac_bits ? min_frac_bits : cur_frac_bits;
    }
  }
  *this->ScaleFracBits<int8_t*>() = min_frac_bits;

  for (int i = 0; i < num_channels; i++) {
    int32_t iscale = (int32_t)((1ll << min_frac_bits) * fscale[i] + 0.5f);
    (*this->Scale<int32_t**>())[i] = iscale;
  }
}

void MliTensorInterface::SetElType(TfLiteType type) {
  if (type == kTfLiteInt8) {
    *this->ElType() = MLI_EL_ASYM_I8;
  } else if (type == kTfLiteInt32) {
    *this->ElType() = MLI_EL_ASYM_I32;
  } else {
    MicroPrintf("Wrong data type. Expected int8_t or int32_t.");
    TFLITE_ABORT;
  }
}
#endif

}  // namespace micro
}  // namespace ops
}  // namespace tflite
