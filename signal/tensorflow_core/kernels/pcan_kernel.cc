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
#include "signal/tensorflow_core/kernels/kernel_utils.h"

namespace tensorflow {
namespace signal {
namespace {

class PcanOp {
 public:
  explicit PcanOp(KernelConstruction* ctx) {
    ctx->GetAttr("snr_shift", &snr_shift_);
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input = ctx->Input(0);
    TensorHandle noise_estimate = ctx->Input(1);
    TensorHandle gain_lut = ctx->Input(2);
    if (!ctx->ok()) {
      return;
    }

    int64_t num_channels = input.NumElements();
    TensorHandle output = ctx->AllocateOutput<uint32_t>(0, {num_channels});
    if (!ctx->ok()) {
      return;
    }

    uint32_t* output_data = output.Data<uint32_t>();
    std::memcpy(output_data, input.Data<uint32_t>(),
                sizeof(uint32_t) * static_cast<size_t>(num_channels));
    tflite::tflm_signal::ApplyPcanAutoGainControlFixed(
        gain_lut.Data<int16_t>(), snr_shift_, noise_estimate.Data<uint32_t>(),
        output_data, static_cast<int32_t>(num_channels));
  }

 private:
  int32_t snr_shift_ = 0;
};

__attribute__((constructor)) void RegisterSignalPcanKernel() {
  RegisterKernel<PcanOp>("SignalPCAN");
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
