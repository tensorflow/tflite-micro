/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <vector>

#include "signal/src/window.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace signal {
namespace {

struct WindowOp {
  int32_t shift;
};

void* WindowOpCreate(TF_OpKernelConstruction* ctx) {
  auto* op = new WindowOp{};
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt32(ctx, "shift", &op->shift, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    delete op;
    op = nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void WindowOpDelete(void* kernel) { delete static_cast<WindowOp*>(kernel); }

void WindowOpCompute(void* kernel, TF_OpKernelContext* ctx) {
  auto* op = static_cast<WindowOp*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;
  TF_Tensor* weight_tensor = nullptr;

  TF_GetInput(ctx, 0, &input_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_GetInput(ctx, 1, &weight_tensor, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    const int num_dims = TF_NumDims(input_tensor);
    std::vector<int64_t> dims(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      dims[i] = TF_Dim(input_tensor, i);
    }

    const int64_t weight_size = TF_TensorElementCount(weight_tensor);
    const int64_t total_elements = TF_TensorElementCount(input_tensor);
    const int64_t outer_dims =
        (weight_size > 0) ? (total_elements / weight_size) : 0;

    TF_Tensor* output_tensor =
        TF_AllocateOutput(ctx, 0, TF_INT16, dims.data(), num_dims,
                          total_elements * sizeof(int16_t), status);
    if (TF_GetCode(status) == TF_OK) {
      const auto* input =
          static_cast<const int16_t*>(TF_TensorData(input_tensor));
      const auto* weights =
          static_cast<const int16_t*>(TF_TensorData(weight_tensor));
      auto* output = static_cast<int16_t*>(TF_TensorData(output_tensor));
      for (int64_t i = 0; i < outer_dims; ++i) {
        tflm_signal::ApplyWindow(&input[i * weight_size], weights,
                                 static_cast<int>(weight_size), op->shift,
                                 &output[i * weight_size]);
      }
      TF_DeleteTensor(output_tensor);
    }
  }

  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
  }
  if (input_tensor != nullptr) {
    TF_DeleteTensor(input_tensor);
  }
  if (weight_tensor != nullptr) {
    TF_DeleteTensor(weight_tensor);
  }
  TF_DeleteStatus(status);
}

// TODO(b/286250473): change back name to "Window" after name clash resolved
__attribute__((constructor)) void RegisterSignalWindowKernel() {
  TF_Status* status = TF_NewStatus();
  auto* builder = TF_NewKernelBuilder("SignalWindow", "CPU", &WindowOpCreate,
                                      &WindowOpCompute, &WindowOpDelete);
  TF_RegisterKernelBuilder("SignalWindow", builder, status);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
