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
#include <vector>

#include "signal/src/circular_buffer.h"
#include "signal/tensorflow_core/kernels/kernel_utils.h"

namespace tensorflow {
namespace signal {
namespace {

class StackerOp {
 public:
  explicit StackerOp(KernelConstruction* ctx) {
    if (!ctx->GetAttr("num_channels", &num_channels_) ||
        !ctx->GetAttr("stacker_left_context", &stacker_left_context_) ||
        !ctx->GetAttr("stacker_right_context", &stacker_right_context_) ||
        !ctx->GetAttr("stacker_step", &stacker_step_)) {
      return;
    }

    buffer_size_ = static_cast<size_t>(num_channels_) *
                   (stacker_left_context_ + stacker_right_context_ + 1);
    step_size_ = static_cast<size_t>(num_channels_) * stacker_step_;

    size_t state_size =
        tflite::tflm_signal::CircularBufferGetNeededMemory(buffer_size_);
    state_.resize(state_size);
    circular_buffer_ = tflite::tflm_signal::CircularBufferInit(
        buffer_size_, state_.data(), state_size);
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input_tensor = ctx->Input(0);
    TensorHandle output_tensor =
        ctx->AllocateOutput<int16_t>(0, {static_cast<int64_t>(buffer_size_)});
    TensorHandle output_valid_tensor = ctx->AllocateScalarOutput<bool>(1);
    if (!ctx->ok()) {
      return;
    }

    const int16_t* input = input_tensor.Data<int16_t>();
    const auto num_channels = static_cast<size_t>(num_channels_);
    tflite::tflm_signal::CircularBufferWrite(circular_buffer_, input,
                                             num_channels);

    // Replicate the first frame left_context times to pad.
    if (!stacker_has_first_frame_) {
      tflite::tflm_signal::CircularBufferExtend(circular_buffer_, num_channels,
                                                stacker_left_context_);
      stacker_has_first_frame_ = true;
    }

    int16_t* output = output_tensor.Data<int16_t>();
    bool* output_valid = output_valid_tensor.Data<bool>();
    if (tflite::tflm_signal::CircularBufferAvailable(circular_buffer_) >=
        buffer_size_) {
      tflite::tflm_signal::CircularBufferGet(circular_buffer_, buffer_size_,
                                             output);
      tflite::tflm_signal::CircularBufferDiscard(circular_buffer_, step_size_);
      *output_valid = true;
    } else {
      *output_valid = false;
    }
  }

 private:
  int32_t num_channels_ = 0;
  int32_t stacker_left_context_ = 0;
  int32_t stacker_right_context_ = 0;
  int32_t stacker_step_ = 0;
  size_t buffer_size_ = 0;
  size_t step_size_ = 0;
  bool stacker_has_first_frame_ = false;
  std::vector<int8_t> state_;
  tflite::tflm_signal::CircularBuffer* circular_buffer_ = nullptr;
};

// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalStackerKernel() {
  RegisterKernel<StackerOp>("SignalStacker");
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
