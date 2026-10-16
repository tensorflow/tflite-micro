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
#include <cstring>

#include "signal/src/pcan_argc_fixed.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace signal {
namespace {

struct PcanOp {
  int32_t snr_shift = 0;
};

void* PcanOpCreate(TF_OpKernelConstruction* ctx) {
  auto* op = new PcanOp{};
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt32(ctx, "snr_shift", &op->snr_shift,
                                       status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    delete op;
    op = nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void PcanOpDelete(void* kernel) { delete static_cast<PcanOp*>(kernel); }

void PcanOpCompute(void* kernel, TF_OpKernelContext* ctx) {
  auto* op = static_cast<PcanOp*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;
  TF_Tensor* noise_tensor = nullptr;
  TF_Tensor* lut_tensor = nullptr;

  TF_GetInput(ctx, 0, &input_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_GetInput(ctx, 1, &noise_tensor, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_GetInput(ctx, 2, &lut_tensor, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    int64_t num_channels = TF_TensorElementCount(input_tensor);
    int64_t out_dims[1] = {num_channels};
    TF_Tensor* output_tensor = TF_AllocateOutput(
        ctx, 0, TF_UINT32, out_dims, 1,
        static_cast<size_t>(num_channels) * sizeof(uint32_t), status);
    if (TF_GetCode(status) == TF_OK) {
      const auto* input =
          static_cast<const uint32_t*>(TF_TensorData(input_tensor));
      const auto* noise_estimate =
          static_cast<const uint32_t*>(TF_TensorData(noise_tensor));
      const auto* gain_lut =
          static_cast<const int16_t*>(TF_TensorData(lut_tensor));
      auto* output = static_cast<uint32_t*>(TF_TensorData(output_tensor));

      std::memcpy(output, input,
                  sizeof(uint32_t) * static_cast<size_t>(num_channels));
      tflite::tflm_signal::ApplyPcanAutoGainControlFixed(
          gain_lut, op->snr_shift, noise_estimate, output,
          static_cast<int32_t>(num_channels));
      TF_DeleteTensor(output_tensor);
    }
  }

  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
  }
  if (input_tensor != nullptr) {
    TF_DeleteTensor(input_tensor);
  }
  if (noise_tensor != nullptr) {
    TF_DeleteTensor(noise_tensor);
  }
  if (lut_tensor != nullptr) {
    TF_DeleteTensor(lut_tensor);
  }
  TF_DeleteStatus(status);
}

__attribute__((constructor)) void RegisterSignalPcanKernel() {
  TF_Status* status = TF_NewStatus();
  auto* builder = TF_NewKernelBuilder("SignalPCAN", "CPU", &PcanOpCreate,
                                      &PcanOpCompute, &PcanOpDelete);
  TF_RegisterKernelBuilder("SignalPCAN", builder, status);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
