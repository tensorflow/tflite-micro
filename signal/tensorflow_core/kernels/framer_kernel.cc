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

class FramerOp : public tensorflow::OpKernel {
 public:
  explicit FramerOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("frame_size", &frame_size_));
    OP_REQUIRES_OK(context, context->GetAttr("frame_step", &frame_step_));
    OP_REQUIRES_OK(context, context->GetAttr("prefill", &prefill_));
    initialized_ = false;
  }

  ~FramerOp() {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input_tensor = context->input(0);
    if (!initialized_) {
      n_frames_ = input_tensor.flat_inner_dims<int16_t>().dimensions().at(1) /
                  frame_step_;
      outer_dims_ = input_tensor.flat_inner_dims<int16_t>().dimensions().at(0);

      state_tensors_.resize(outer_dims_);
      circular_buffers_.resize(outer_dims_);

      // Calculate the capacity of the circular buffer. Round up the frame size
      // to
      // a multiple of frame step. Saves memory relative to the simpler
      // frame_size + frame_step. For example:
      // step_size = 160, frame_size = 400
      // capacity = 480 vs. step_size + frame_size = 560
      size_t capacity =
          (frame_size_ + frame_step_ - 1) / frame_step_ * frame_step_;
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
        if (prefill_) {
          tflite::tflm_signal::CircularBufferWriteZeros(
              circular_buffers_[i], frame_size_ - frame_step_);
        }
      }

      initialized_ = true;
    }

    // Split the last dimension of the input into {n_frames_, frame_size_}.
    TensorShape output_shape = input_tensor.shape();
    output_shape.AddDim(frame_size_);
    output_shape.set_dim(output_shape.dims() - 2, n_frames_);

    tensorflow::Tensor* output_tensor = nullptr;
    tensorflow::Tensor* output_valid_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, {}, &output_valid_tensor));

    bool output_valid = true;
    for (int i = 0; i < outer_dims_; i++) {
      for (int frame = 0; frame < n_frames_; frame++) {
        int input_index = (i * n_frames_ + frame) * frame_step_;
        int output_index = (i * n_frames_ + frame) * frame_size_;
        tflite::tflm_signal::CircularBufferWrite(
            circular_buffers_[i],
            &(input_tensor.flat<int16_t>().data())[input_index], frame_step_);
        if (tflite::tflm_signal::CircularBufferAvailable(
                circular_buffers_[i]) >= (unsigned)frame_size_) {
          tflite::tflm_signal::CircularBufferGet(
              circular_buffers_[i], frame_size_,
              &(reinterpret_cast<int16_t*>(
                  output_tensor->data()))[output_index]);
          tflite::tflm_signal::CircularBufferDiscard(circular_buffers_[i],
                                                     frame_step_);
        } else {
          output_valid = false;
        }
      }
      *output_valid_tensor->flat<bool>().data() = output_valid;
    }
  }

 private:
  bool initialized_;
  int frame_size_;
  int frame_step_;
  int outer_dims_;
  bool prefill_;
  int n_frames_;
  std::vector<Tensor> state_tensors_;
  std::vector<struct tflite::tflm_signal::CircularBuffer*> circular_buffers_;
};

// TODO(b/286250473): change back name after name clash resolved
REGISTER_KERNEL_BUILDER(Name("SignalFramer").Device(tensorflow::DEVICE_CPU),
                        FramerOp);

}  // namespace signal
}  // namespace tensorflow
