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

#include "signal/src/overlap_add.h"
#include "signal/tensorflow_core/kernels/kernel_utils.h"
#include "tensorflow/c/tf_datatype.h"

namespace tensorflow {
namespace signal {
namespace {

template <typename T>
class OverlapAddOp {
 public:
  explicit OverlapAddOp(KernelConstruction* ctx) {
    ctx->GetAttr("frame_step", &frame_step_);
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input_tensor = ctx->Input(0);
    if (!ctx->ok()) {
      return;
    }

    const std::vector<int64_t> in_dims = input_tensor.Shape();
    if (!initialized_ && in_dims.size() >= 2) {
      frame_size_ = in_dims[in_dims.size() - 1];
      n_frames_ = in_dims[in_dims.size() - 2];
      int64_t inner = n_frames_ * frame_size_;
      outer_dims_ = (inner > 0) ? (input_tensor.NumElements() / inner) : 0;

      state_buffers_.assign(
          static_cast<size_t>(outer_dims_),
          std::vector<T>(static_cast<size_t>(frame_size_), T{0}));
      initialized_ = true;
    }

    std::vector<int64_t> out_dims;
    if (!in_dims.empty()) {
      out_dims.assign(in_dims.begin(), in_dims.end() - 1);
      if (!out_dims.empty()) {
        out_dims.back() = n_frames_ * frame_step_;
      }
    }

    TensorHandle output_tensor = ctx->AllocateOutput<T>(0, out_dims);
    if (!ctx->ok()) {
      return;
    }

    const T* input = input_tensor.Data<T>();
    T* output = output_tensor.Data<T>();
    for (int64_t i = 0; i < outer_dims_; ++i) {
      T* buffer = state_buffers_[static_cast<size_t>(i)].data();
      for (int64_t frame = 0; frame < n_frames_; ++frame) {
        int64_t input_index = (i * n_frames_ + frame) * frame_size_;
        int64_t output_index = (i * n_frames_ + frame) * frame_step_;
        tflm_signal::OverlapAdd(&input[input_index], buffer,
                                static_cast<int>(frame_size_),
                                &output[output_index], frame_step_);
      }
    }
  }

 private:
  int64_t frame_size_ = 0;
  int32_t frame_step_ = 0;
  int64_t n_frames_ = 0;
  int64_t outer_dims_ = 0;
  bool initialized_ = false;
  std::vector<std::vector<T>> state_buffers_;
};

// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalOverlapAddKernels() {
  RegisterTypedKernel<OverlapAddOp<float>>("SignalOverlapAdd", TF_FLOAT);
  RegisterTypedKernel<OverlapAddOp<int16_t>>("SignalOverlapAdd", TF_INT16);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
