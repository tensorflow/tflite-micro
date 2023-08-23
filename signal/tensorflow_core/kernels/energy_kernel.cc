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

#include "signal/src/complex.h"
#include "signal/src/energy.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace signal {

class EnergyOp : public tensorflow::OpKernel {
 public:
  explicit EnergyOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("start_index", &start_index_));
    OP_REQUIRES_OK(context, context->GetAttr("end_index", &end_index_));
  }
  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input_tensor = context->input(0);
    const int16_t* input = input_tensor.flat<int16_t>().data();
    tensorflow::Tensor* output_tensor = nullptr;
    // The input is complex. The output is real.
    int output_size = input_tensor.flat<int16>().size() >> 1;

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {output_size}, &output_tensor));
    uint32* output = output_tensor->flat<uint32>().data();

    tflite::tflm_signal::SpectrumToEnergy(
        reinterpret_cast<const Complex<int16_t>*>(input), start_index_,
        end_index_, output);
  }

 private:
  int start_index_;
  int end_index_;
};

// TODO(b/286250473): change back name after name clash resolved
REGISTER_KERNEL_BUILDER(Name("SignalEnergy").Device(tensorflow::DEVICE_CPU),
                        EnergyOp);

}  // namespace signal
}  // namespace tensorflow
