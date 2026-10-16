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

class FramerOp {
 public:
  explicit FramerOp(KernelConstruction* ctx) {
    ctx->GetAttr("frame_size", &frame_size_);
    ctx->GetAttr("frame_step", &frame_step_);
    ctx->GetAttr("prefill", &prefill_);
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input_tensor = ctx->Input(0);
    if (!ctx->ok()) {
      return;
    }

    const std::vector<int64_t> in_dims = input_tensor.Shape();
    const int64_t last_dim = in_dims.empty() ? 0 : in_dims.back();
    const int64_t total_elements = input_tensor.NumElements();

    if (!initialized_) {
      n_frames_ = last_dim / frame_step_;
      outer_dims_ = (last_dim > 0) ? (total_elements / last_dim) : 0;

      state_buffers_.resize(static_cast<size_t>(outer_dims_));
      circular_buffers_.resize(static_cast<size_t>(outer_dims_));

      // Round up frame_size to a multiple of frame_step to save state memory.
      size_t capacity =
          static_cast<size_t>((frame_size_ + frame_step_ - 1) / frame_step_) *
          frame_step_;
      size_t state_size =
          tflite::tflm_signal::CircularBufferGetNeededMemory(capacity);
      for (int64_t i = 0; i < outer_dims_; ++i) {
        auto idx = static_cast<size_t>(i);
        state_buffers_[idx].resize(state_size);
        circular_buffers_[idx] = tflite::tflm_signal::CircularBufferInit(
            capacity, state_buffers_[idx].data(), state_size);
        if (prefill_) {
          tflite::tflm_signal::CircularBufferWriteZeros(
              circular_buffers_[idx], frame_size_ - frame_step_);
        }
      }
      initialized_ = true;
    }

    // Split the last dimension of the input into {n_frames, frame_size}.
    std::vector<int64_t> out_dims = in_dims;
    if (out_dims.empty()) {
      out_dims = {n_frames_, frame_size_};
    } else {
      out_dims.back() = n_frames_;
      out_dims.push_back(frame_size_);
    }

    TensorHandle output_tensor = ctx->AllocateOutput<int16_t>(0, out_dims);
    TensorHandle output_valid_tensor = ctx->AllocateScalarOutput<bool>(1);
    if (!ctx->ok()) {
      return;
    }

    const int16_t* input = input_tensor.Data<int16_t>();
    int16_t* output = output_tensor.Data<int16_t>();
    bool* out_valid = output_valid_tensor.Data<bool>();

    bool output_valid = true;
    const auto frame_step = static_cast<size_t>(frame_step_);
    const auto frame_size = static_cast<size_t>(frame_size_);

    for (int64_t i = 0; i < outer_dims_; ++i) {
      auto* cb = circular_buffers_[static_cast<size_t>(i)];
      for (int64_t frame = 0; frame < n_frames_; ++frame) {
        int64_t input_index = (i * n_frames_ + frame) * frame_step_;
        int64_t output_index = (i * n_frames_ + frame) * frame_size_;
        tflite::tflm_signal::CircularBufferWrite(cb, &input[input_index],
                                                 frame_step);
        if (tflite::tflm_signal::CircularBufferAvailable(cb) >= frame_size) {
          tflite::tflm_signal::CircularBufferGet(cb, frame_size,
                                                 &output[output_index]);
          tflite::tflm_signal::CircularBufferDiscard(cb, frame_step);
        } else {
          output_valid = false;
        }
      }
    }
    *out_valid = output_valid;
  }

 private:
  bool initialized_ = false;
  int32_t frame_size_ = 0;
  int32_t frame_step_ = 0;
  int64_t outer_dims_ = 0;
  bool prefill_ = false;
  int64_t n_frames_ = 0;
  std::vector<std::vector<int8_t>> state_buffers_;
  std::vector<tflite::tflm_signal::CircularBuffer*> circular_buffers_;
};

// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalFramerKernel() {
  RegisterKernel<FramerOp>("SignalFramer");
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
