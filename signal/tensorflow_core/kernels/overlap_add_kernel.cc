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

#include "signal/src/overlap_add.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace signal {

template <typename T, DataType E>
class OverlapAddOp : public tensorflow::OpKernel {
 public:
  explicit OverlapAddOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("frame_step", &frame_step_));
    initialized_ = false;
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input_tensor = context->input(0);
    if (!initialized_) {
      outer_dims_ = input_tensor.flat_inner_dims<T, 3>().dimensions().at(0);
      n_frames_ = input_tensor.flat_inner_dims<T, 3>().dimensions().at(1);
      frame_size_ = input_tensor.flat_inner_dims<T, 3>().dimensions().at(2);

      state_tensors_.resize(outer_dims_);
      for (int i = 0; i < outer_dims_; i++) {
        OP_REQUIRES_OK(context,
                       context->allocate_temp(
                           E, TensorShape({static_cast<int32_t>(frame_size_)}),
                           &state_tensors_[i]));
        memset(state_tensors_[i].flat<T>().data(), 0, sizeof(T) * frame_size_);
      }
      initialized_ = true;
    }

    TensorShape output_shape = input_tensor.shape();
    output_shape.RemoveDim(output_shape.dims() - 1);
    output_shape.set_dim(output_shape.dims() - 1, n_frames_ * frame_step_);

    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    const T* input = input_tensor.flat<T>().data();
    T* output = output_tensor->flat<T>().data();
    for (int i = 0; i < outer_dims_; i++) {
      T* buffer = state_tensors_[i].flat<T>().data();
      for (int frame = 0; frame < n_frames_; frame++) {
        int input_index = (i * n_frames_ + frame) * frame_size_;
        int output_index = (i * n_frames_ + frame) * frame_step_;
        tflm_signal::OverlapAdd(&input[input_index], buffer, frame_size_,
                                &output[output_index], frame_step_);
      }
    }
  }

 private:
  int frame_size_;
  int frame_step_;
  int n_frames_;
  int outer_dims_;
  bool initialized_;
  std::vector<Tensor> state_tensors_;
};

// TODO(b/286250473): change back name after name clash resolved
REGISTER_KERNEL_BUILDER(Name("SignalOverlapAdd")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        OverlapAddOp<float, DT_FLOAT>);
REGISTER_KERNEL_BUILDER(Name("SignalOverlapAdd")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int16_t>("T"),
                        OverlapAddOp<int16_t, DT_INT16>);

}  // namespace signal
}  // namespace tensorflow
