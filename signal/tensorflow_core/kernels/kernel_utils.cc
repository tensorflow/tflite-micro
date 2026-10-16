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

#include "signal/tensorflow_core/kernels/kernel_utils.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensorflow {
namespace signal {

std::vector<int64_t> TensorHandle::Shape() const {
  const int num_dims = NumDims();
  std::vector<int64_t> dims(static_cast<size_t>(num_dims));
  for (int i = 0; i < num_dims; ++i) {
    dims[static_cast<size_t>(i)] = Dim(i);
  }
  return dims;
}

KernelConstruction::~KernelConstruction() {
  if (!ok()) {
    TF_OpKernelConstruction_Failure(ctx_, status_);
  }
  TF_DeleteStatus(status_);
}

bool KernelConstruction::GetAttr(const char* name, int32_t* value) {
  if (ok()) {
    TF_OpKernelConstruction_GetAttrInt32(ctx_, name, value, status_);
  }
  return ok();
}

bool KernelConstruction::GetAttr(const char* name, bool* value) {
  if (ok()) {
    TF_Bool raw_bool = 0;
    TF_OpKernelConstruction_GetAttrBool(ctx_, name, &raw_bool, status_);
    if (ok()) {
      *value = (raw_bool != 0);
    }
  }
  return ok();
}

KernelContext::~KernelContext() {
  if (!ok()) {
    TF_OpKernelContext_Failure(ctx_, status_);
  }
  TF_DeleteStatus(status_);
}

TensorHandle KernelContext::Input(int index) {
  TF_Tensor* tensor = nullptr;
  if (ok()) {
    TF_GetInput(ctx_, index, &tensor, status_);
  }
  return TensorHandle(tensor);
}

TensorHandle KernelContext::AllocateOutputRaw(int index, TF_DataType dtype,
                                              const int64_t* dims, int num_dims,
                                              size_t byte_len) {
  if (!ok()) {
    return TensorHandle(nullptr);
  }
  return TensorHandle(
      TF_AllocateOutput(ctx_, index, dtype, dims, num_dims, byte_len, status_));
}

void RegisterKernelBuilderRaw(const char* op_name,
                              void* (*create_fn)(TF_OpKernelConstruction*),
                              void (*compute_fn)(void*, TF_OpKernelContext*),
                              void (*delete_fn)(void*),
                              bool has_type_constraint,
                              TF_DataType type_constraint) {
  TF_Status* status = TF_NewStatus();
  auto* builder =
      TF_NewKernelBuilder(op_name, "CPU", create_fn, compute_fn, delete_fn);
  if (has_type_constraint) {
    TF_KernelBuilder_TypeConstraint(builder, "T", type_constraint, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(op_name, builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  TF_DeleteStatus(status);
}

}  // namespace signal
}  // namespace tensorflow
