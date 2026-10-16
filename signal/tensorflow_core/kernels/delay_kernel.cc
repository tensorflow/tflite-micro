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

class DelayOp {
 public:
  explicit DelayOp(KernelConstruction* ctx) {
    ctx->GetAttr("delay_length", &delay_length_);
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input_tensor = ctx->Input(0);
    if (!ctx->ok()) {
      return;
    }

    const std::vector<int64_t> dims = input_tensor.Shape();
    const int64_t total_elements = input_tensor.NumElements();
    if (!initialized_) {
      frame_size_ = dims.empty() ? 1 : dims.back();
      outer_dims_ = (frame_size_ > 0) ? (total_elements / frame_size_) : 0;

      state_buffers_.resize(static_cast<size_t>(outer_dims_));
      circular_buffers_.resize(static_cast<size_t>(outer_dims_));

      size_t capacity = static_cast<size_t>(frame_size_ + delay_length_);
      size_t state_size =
          tflite::tflm_signal::CircularBufferGetNeededMemory(capacity);
      for (int64_t i = 0; i < outer_dims_; ++i) {
        auto idx = static_cast<size_t>(i);
        state_buffers_[idx].resize(state_size);
        circular_buffers_[idx] = tflite::tflm_signal::CircularBufferInit(
            capacity, state_buffers_[idx].data(), state_size);
        tflite::tflm_signal::CircularBufferWriteZeros(circular_buffers_[idx],
                                                      delay_length_);
      }
      initialized_ = true;
    }

    TensorHandle output_tensor = ctx->AllocateOutput<int16_t>(0, dims);
    if (!ctx->ok()) {
      return;
    }

    const int16_t* input = input_tensor.Data<int16_t>();
    int16_t* output = output_tensor.Data<int16_t>();
    const auto frame_size = static_cast<size_t>(frame_size_);

    for (int64_t dim_index = 0, sample_index = 0; dim_index < outer_dims_;
         ++dim_index, sample_index += frame_size_) {
      auto* cb = circular_buffers_[static_cast<size_t>(dim_index)];
      tflite::tflm_signal::CircularBufferWrite(cb, &input[sample_index],
                                               frame_size);
      tflite::tflm_signal::CircularBufferGet(cb, frame_size,
                                             &output[sample_index]);
      tflite::tflm_signal::CircularBufferDiscard(cb, frame_size);
    }
  }

 private:
  bool initialized_ = false;
  int64_t frame_size_ = 0;
  int32_t delay_length_ = 0;
  int64_t outer_dims_ = 0;
  std::vector<std::vector<int8_t>> state_buffers_;
  std::vector<tflite::tflm_signal::CircularBuffer*> circular_buffers_;
};

// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalDelayKernel() {
  RegisterKernel<DelayOp>("SignalDelay");
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
