/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>

#include "signal/src/complex.h"
#include "signal/src/energy.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace signal {
namespace {

struct EnergyOp {
  int32_t start_index = 0;
  int32_t end_index = 0;
};

void* EnergyOpCreate(TF_OpKernelConstruction* ctx) {
  auto* op = new EnergyOp{};
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt32(ctx, "start_index", &op->start_index,
                                       status);
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrInt32(ctx, "end_index", &op->end_index,
                                         status);
  }
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    delete op;
    op = nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void EnergyOpDelete(void* kernel) { delete static_cast<EnergyOp*>(kernel); }

void EnergyOpCompute(void* kernel, TF_OpKernelContext* ctx) {
  auto* op = static_cast<EnergyOp*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;

  TF_GetInput(ctx, 0, &input_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    // The input is complex. The output is real.
    int64_t output_size = TF_TensorElementCount(input_tensor) >> 1;
    int64_t out_dims[1] = {output_size};
    TF_Tensor* output_tensor = TF_AllocateOutput(
        ctx, 0, TF_UINT32, out_dims, 1,
        static_cast<size_t>(output_size) * sizeof(uint32_t), status);
    if (TF_GetCode(status) == TF_OK) {
      const auto* input =
          static_cast<const int16_t*>(TF_TensorData(input_tensor));
      auto* output = static_cast<uint32_t*>(TF_TensorData(output_tensor));
      tflite::tflm_signal::SpectrumToEnergy(
          reinterpret_cast<const Complex<int16_t>*>(input), op->start_index,
          op->end_index, output);
      TF_DeleteTensor(output_tensor);
    }
  }

  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
  }
  if (input_tensor != nullptr) {
    TF_DeleteTensor(input_tensor);
  }
  TF_DeleteStatus(status);
}

// TODO(b/286250473): change back name after name clash resolved
void RegisterSignalEnergyKernel() {
  TF_Status* status = TF_NewStatus();
  auto* builder = TF_NewKernelBuilder("SignalEnergy", "CPU", &EnergyOpCreate,
                                      &EnergyOpCompute, &EnergyOpDelete);
  TF_RegisterKernelBuilder("SignalEnergy", builder, status);
  TF_DeleteStatus(status);
}

const bool kSignalEnergyKernelRegistered = []() {
  RegisterSignalEnergyKernel();
  return true;
}();

}  // namespace
}  // namespace signal
}  // namespace tensorflow
