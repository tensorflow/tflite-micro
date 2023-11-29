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

#include "signal/src/circular_buffer.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace signal {

class DelayOp : public tensorflow::OpKernel {
 public:
  explicit DelayOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("delay_length", &delay_length_));
    initialized_ = false;
  }

  ~DelayOp() {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input_tensor = context->input(0);
    if (!initialized_) {
      frame_size_ = input_tensor.flat_inner_dims<int16_t>().dimensions().at(1);
      outer_dims_ = input_tensor.flat_inner_dims<int16_t>().dimensions().at(0);

      state_tensors_.resize(outer_dims_);
      circular_buffers_.resize(outer_dims_);

      // Calculate the capacity of the circular buffer.
      size_t capacity = frame_size_ + delay_length_;
      size_t state_size =
          tflite::tflm_signal::CircularBufferGetNeededMemory(capacity);
      for (int i = 0; i < outer_dims_; i++) {
        OP_REQUIRES_OK(
            context,
            context->allocate_temp(
                DT_INT8, TensorShape({static_cast<int32_t>(state_size)}),
                &state_tensors_[i]));
        int8_t* state_ = state_tensors_[i].flat<int8_t>().data();
        circular_buffers_[i] = tflite::tflm_signal::CircularBufferInit(
            capacity, state_, state_size);
        tflite::tflm_signal::CircularBufferWriteZeros(circular_buffers_[i],
                                                      delay_length_);
      }
      initialized_ = true;
    }

    TensorShape output_shape = input_tensor.shape();
    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    for (int dim_index = 0, sample_index = 0; dim_index < outer_dims_;
         dim_index++, sample_index += frame_size_) {
      tflite::tflm_signal::CircularBufferWrite(
          circular_buffers_[dim_index],
          &input_tensor.flat<int16_t>().data()[sample_index], frame_size_);
      tflite::tflm_signal::CircularBufferGet(
          circular_buffers_[dim_index], frame_size_,
          &(reinterpret_cast<int16_t*>(output_tensor->data()))[sample_index]);
      tflite::tflm_signal::CircularBufferDiscard(circular_buffers_[dim_index],
                                                 frame_size_);
    }
  }

 private:
  bool initialized_;
  int frame_size_;
  int delay_length_;
  int outer_dims_;
  std::vector<Tensor> state_tensors_;
  std::vector<struct tflite::tflm_signal::CircularBuffer*> circular_buffers_;
};

// TODO(b/286250473): change back name after name clash resolved
REGISTER_KERNEL_BUILDER(Name("SignalDelay").Device(tensorflow::DEVICE_CPU),
                        DelayOp);

}  // namespace signal
}  // namespace tensorflow
