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

#ifndef SIGNAL_TENSORFLOW_CORE_KERNELS_KERNEL_UTILS_H_
#define SIGNAL_TENSORFLOW_CORE_KERNELS_KERNEL_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace signal {

template <typename T>
constexpr TF_DataType DataTypeOf();

template <>
constexpr TF_DataType DataTypeOf<float>() {
  return TF_FLOAT;
}
template <>
constexpr TF_DataType DataTypeOf<int16_t>() {
  return TF_INT16;
}
template <>
constexpr TF_DataType DataTypeOf<int32_t>() {
  return TF_INT32;
}
template <>
constexpr TF_DataType DataTypeOf<uint32_t>() {
  return TF_UINT32;
}
template <>
constexpr TF_DataType DataTypeOf<uint64_t>() {
  return TF_UINT64;
}
template <>
constexpr TF_DataType DataTypeOf<bool>() {
  return TF_BOOL;
}

// Move-only RAII wrapper for TF_Tensor*.
class TensorHandle {
 public:
  TensorHandle() = default;
  explicit TensorHandle(TF_Tensor* tensor) : tensor_(tensor) {}
  ~TensorHandle() {
    if (tensor_ != nullptr) {
      TF_DeleteTensor(tensor_);
    }
  }

  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;

  TensorHandle(TensorHandle&& other) noexcept : tensor_(other.tensor_) {
    other.tensor_ = nullptr;
  }
  TensorHandle& operator=(TensorHandle&& other) noexcept {
    if (this != &other) {
      if (tensor_ != nullptr) {
        TF_DeleteTensor(tensor_);
      }
      tensor_ = other.tensor_;
      other.tensor_ = nullptr;
    }
    return *this;
  }

  bool valid() const { return tensor_ != nullptr; }
  int NumDims() const { return TF_NumDims(tensor_); }
  int64_t Dim(int index) const { return TF_Dim(tensor_, index); }
  int64_t NumElements() const { return TF_TensorElementCount(tensor_); }

  std::vector<int64_t> Shape() const;

  template <typename T>
  const T* Data() const {
    return static_cast<const T*>(TF_TensorData(tensor_));
  }

  template <typename T>
  T* Data() {
    return static_cast<T*>(TF_TensorData(tensor_));
  }

 private:
  TF_Tensor* tensor_ = nullptr;
};

// C++ RAII wrapper around TF_OpKernelConstruction*.
class KernelConstruction {
 public:
  explicit KernelConstruction(TF_OpKernelConstruction* ctx)
      : ctx_(ctx), status_(TF_NewStatus()) {}

  ~KernelConstruction();

  KernelConstruction(const KernelConstruction&) = delete;
  KernelConstruction& operator=(const KernelConstruction&) = delete;

  bool ok() const { return TF_GetCode(status_) == TF_OK; }

  bool GetAttr(const char* name, int32_t* value);
  bool GetAttr(const char* name, bool* value);

 private:
  TF_OpKernelConstruction* ctx_;
  TF_Status* status_;
};

// C++ RAII wrapper around TF_OpKernelContext*.
class KernelContext {
 public:
  explicit KernelContext(TF_OpKernelContext* ctx)
      : ctx_(ctx), status_(TF_NewStatus()) {}

  ~KernelContext();

  KernelContext(const KernelContext&) = delete;
  KernelContext& operator=(const KernelContext&) = delete;

  bool ok() const { return TF_GetCode(status_) == TF_OK; }

  TensorHandle Input(int index);

  TensorHandle AllocateOutputRaw(int index, TF_DataType dtype,
                                 const int64_t* dims, int num_dims,
                                 size_t byte_len);

  template <typename T>
  TensorHandle AllocateOutput(int index, const std::vector<int64_t>& dims) {
    size_t num_elements = 1;
    for (int64_t dim : dims) {
      num_elements *= static_cast<size_t>(dim);
    }
    return AllocateOutputRaw(index, DataTypeOf<T>(), dims.data(),
                             static_cast<int>(dims.size()),
                             num_elements * sizeof(T));
  }

  template <typename T>
  TensorHandle AllocateScalarOutput(int index) {
    return AllocateOutputRaw(index, DataTypeOf<T>(), nullptr, 0, sizeof(T));
  }

 private:
  TF_OpKernelContext* ctx_;
  TF_Status* status_;
};

void RegisterKernelBuilderRaw(const char* op_name,
                              void* (*create_fn)(TF_OpKernelConstruction*),
                              void (*compute_fn)(void*, TF_OpKernelContext*),
                              void (*delete_fn)(void*),
                              bool has_type_constraint,
                              TF_DataType type_constraint);

template <typename Kernel>
void* CreateKernelInstance(TF_OpKernelConstruction* raw_ctx) {
  KernelConstruction ctx(raw_ctx);
  auto* kernel = new Kernel(&ctx);
  if (!ctx.ok()) {
    delete kernel;
    return nullptr;
  }
  return kernel;
}

template <typename Kernel>
void ComputeKernelInstance(void* kernel, TF_OpKernelContext* raw_ctx) {
  KernelContext ctx(raw_ctx);
  static_cast<Kernel*>(kernel)->Compute(&ctx);
}

template <typename Kernel>
void DeleteKernelInstance(void* kernel) {
  delete static_cast<Kernel*>(kernel);
}

template <typename Kernel>
void RegisterKernel(const char* op_name) {
  RegisterKernelBuilderRaw(op_name, &CreateKernelInstance<Kernel>,
                           &ComputeKernelInstance<Kernel>,
                           &DeleteKernelInstance<Kernel>,
                           /*has_type_constraint=*/false, TF_FLOAT);
}

template <typename Kernel>
void RegisterTypedKernel(const char* op_name, TF_DataType type) {
  RegisterKernelBuilderRaw(op_name, &CreateKernelInstance<Kernel>,
                           &ComputeKernelInstance<Kernel>,
                           &DeleteKernelInstance<Kernel>,
                           /*has_type_constraint=*/true, type);
}

}  // namespace signal
}  // namespace tensorflow

#endif  // SIGNAL_TENSORFLOW_CORE_KERNELS_KERNEL_UTILS_H_
