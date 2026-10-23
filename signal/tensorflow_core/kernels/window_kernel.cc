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

#include "signal/src/window.h"
#include "signal/tensorflow_core/kernels/kernel_utils.h"

namespace tensorflow {
namespace signal {
namespace {

class WindowOp {
 public:
  explicit WindowOp(KernelConstruction* ctx) { ctx->GetAttr("shift", &shift_); }

  void Compute(KernelContext* ctx) {
    TensorHandle input_tensor = ctx->Input(0);
    TensorHandle weight_tensor = ctx->Input(1);
    if (!ctx->ok()) {
      return;
    }

    int64_t weight_size = weight_tensor.NumElements();
    int64_t outer_dims =
        (weight_size > 0) ? (input_tensor.NumElements() / weight_size) : 0;

    TensorHandle output_tensor =
        ctx->AllocateOutput<int16_t>(0, input_tensor.Shape());
    if (!ctx->ok()) {
      return;
    }

    const int16_t* input = input_tensor.Data<int16_t>();
    const int16_t* weights = weight_tensor.Data<int16_t>();
    int16_t* output = output_tensor.Data<int16_t>();
    for (int64_t i = 0; i < outer_dims; ++i) {
      tflm_signal::ApplyWindow(&input[i * weight_size], weights,
                               static_cast<int>(weight_size), shift_,
                               &output[i * weight_size]);
    }
  }

 private:
  int32_t shift_ = 0;
};

// TODO(b/286250473): change back name to "Window" after name clash resolved
__attribute__((constructor)) void RegisterSignalWindowKernel() {
  RegisterKernel<WindowOp>("SignalWindow");
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
