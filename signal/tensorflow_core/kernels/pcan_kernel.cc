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

#include "signal/src/pcan_argc_fixed.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace signal {

class PcanOp : public tensorflow::OpKernel {
 public:
  explicit PcanOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("snr_shift", &snr_shift_));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    tensorflow::Tensor* output_tensor = nullptr;
    const uint32_t* input = context->input(0).flat<uint32_t>().data();
    const uint32_t* noise_estimate = context->input(1).flat<uint32_t>().data();
    const int16_t* gain_lut = context->input(2).flat<int16_t>().data();
    int32_t num_channels = context->input(0).NumElements();
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {num_channels}, &output_tensor));
    uint32_t* output = output_tensor->flat<uint32_t>().data();

    memcpy(output, input, sizeof(uint32_t) * num_channels);
    tflite::tflm_signal::ApplyPcanAutoGainControlFixed(
        gain_lut, snr_shift_, noise_estimate, output, num_channels);
  }

 private:
  int snr_shift_;
};

REGISTER_KERNEL_BUILDER(Name("SignalPCAN").Device(tensorflow::DEVICE_CPU),
                        PcanOp);

}  // namespace signal
}  // namespace tensorflow
