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

#include "signal/lite/kernels/kernel_util.h"

#include <stddef.h>
#include <stdint.h>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace tflm_signal {
namespace lite {

int GetFlatSize(const TfLiteOpaqueTensor* tensor) {
  int flat_size = 1;
  int num_dims = TfLiteOpaqueTensorNumDims(tensor);
  for (int i = 0; i < num_dims; ++i) {
    flat_size *= TfLiteOpaqueTensorDim(tensor, i);
  }
  return flat_size;
}

flexbuffers::Map GetFlexbufferMap(const char* buffer, size_t length) {
  if (buffer == nullptr || length == 0) {
    return flexbuffers::Map::EmptyMap();
  }
  return flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
      .AsMap();
}

TfLiteOperator* MakeOperator(
    const char* custom_name,
    void* (*init)(void*, TfLiteOpaqueContext*, const char*, size_t),
    void (*free_fn)(void*, TfLiteOpaqueContext*, void*),
    TfLiteStatus (*prepare)(void*, TfLiteOpaqueContext*, TfLiteOpaqueNode*),
    TfLiteStatus (*invoke)(void*, TfLiteOpaqueContext*, TfLiteOpaqueNode*)) {
  TfLiteOperator* op =
      TfLiteOperatorCreate(kTfLiteBuiltinCustom, custom_name, /*version=*/1,
                           /*user_data=*/nullptr);
  if (init != nullptr) {
    TfLiteOperatorSetInitWithData(op, init);
  }
  if (free_fn != nullptr) {
    TfLiteOperatorSetFreeWithData(op, free_fn);
  }
  TfLiteOperatorSetPrepareWithData(op, prepare);
  TfLiteOperatorSetInvokeWithData(op, invoke);
  return op;
}

::TfLiteRegistration MakeRegistration(const char* custom_name,
                                      TfLiteOperator* op) {
  ::TfLiteRegistration reg{};
  reg.builtin_code = kTfLiteBuiltinCustom;
  reg.custom_name = custom_name;
  reg.version = 1;
  reg.registration_external = op;
  return reg;
}

}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
