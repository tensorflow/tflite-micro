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

#include <cstdint>

#include "signal/src/complex.h"
#include "signal/src/energy.h"
#include "signal/tensorflow_core/kernels/kernel_utils.h"

namespace tensorflow {
namespace signal {
namespace {

class EnergyOp {
 public:
  explicit EnergyOp(KernelConstruction* ctx) {
    ctx->GetAttr("start_index", &start_index_);
    ctx->GetAttr("end_index", &end_index_);
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input_tensor = ctx->Input(0);
    if (!ctx->ok()) {
      return;
    }

    // The input is complex (interleaved real/imag pairs); the output is real.
    int64_t output_size = input_tensor.NumElements() / 2;
    TensorHandle output_tensor =
        ctx->AllocateOutput<uint32_t>(0, {output_size});
    if (!ctx->ok()) {
      return;
    }

    const auto* input =
        reinterpret_cast<const Complex<int16_t>*>(input_tensor.Data<int16_t>());
    uint32_t* output = output_tensor.Data<uint32_t>();
    tflite::tflm_signal::SpectrumToEnergy(input, start_index_, end_index_,
                                          output);
  }

 private:
  int32_t start_index_ = 0;
  int32_t end_index_ = 0;
};

// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalEnergyKernel() {
  RegisterKernel<EnergyOp>("SignalEnergy");
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
