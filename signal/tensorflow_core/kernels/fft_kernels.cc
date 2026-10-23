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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "signal/src/fft_auto_scale.h"
#include "signal/src/irfft.h"
#include "signal/src/rfft.h"
#include "signal/tensorflow_core/kernels/kernel_utils.h"
#include "tensorflow/c/tf_datatype.h"

namespace tensorflow {
namespace signal {
namespace {

template <typename T, size_t (*GetNeededMemoryFn)(int32_t),
          void* (*InitFn)(int32_t, void*, size_t),
          void (*ApplyFn)(void*, const T* input, Complex<T>*)>
class RfftOp {
 public:
  explicit RfftOp(KernelConstruction* ctx) {
    if (!ctx->GetAttr("fft_length", &fft_length_)) {
      return;
    }
    work_area_.resize(static_cast<size_t>(fft_length_));
    // Each complex coefficient occupies two scalar elements.
    subband_array_size_ = ((fft_length_ / 2) + 1) * 2;
    size_t state_size = GetNeededMemoryFn(fft_length_);
    state_.resize(state_size);
    InitFn(fft_length_, state_.data(), state_size);
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input_tensor = ctx->Input(0);
    if (!ctx->ok()) {
      return;
    }

    std::vector<int64_t> out_dims = input_tensor.Shape();
    int64_t frame_size = out_dims.empty() ? 0 : out_dims.back();
    int64_t outer_dims =
        (frame_size > 0) ? (input_tensor.NumElements() / frame_size) : 0;
    if (!out_dims.empty()) {
      out_dims.back() = subband_array_size_;
    }

    TensorHandle output_tensor = ctx->AllocateOutput<T>(0, out_dims);
    if (!ctx->ok()) {
      return;
    }

    const T* input = input_tensor.Data<T>();
    T* output = output_tensor.Data<T>();
    for (int64_t i = 0; i < outer_dims; ++i) {
      auto input_in_work_end =
          std::copy_n(&input[i * frame_size], frame_size, work_area_.data());
      // Zero-pad input to FFT length.
      std::fill(input_in_work_end, work_area_.data() + fft_length_, T{0});
      ApplyFn(state_.data(), work_area_.data(),
              reinterpret_cast<Complex<T>*>(&output[i * subband_array_size_]));
    }
  }

 private:
  int32_t fft_length_ = 0;
  int32_t subband_array_size_ = 0;
  std::vector<T> work_area_;
  std::vector<int8_t> state_;
};

template <typename T, size_t (*GetNeededMemoryFn)(int32_t),
          void* (*InitFn)(int32_t, void*, size_t),
          void (*ApplyFn)(void*, const Complex<T>* input, T*)>
class IrfftOp {
 public:
  explicit IrfftOp(KernelConstruction* ctx) {
    if (!ctx->GetAttr("fft_length", &fft_length_)) {
      return;
    }
    subband_array_size_ = ((fft_length_ / 2) + 1) * 2;
    size_t state_size = GetNeededMemoryFn(fft_length_);
    state_.resize(state_size);
    InitFn(fft_length_, state_.data(), state_size);
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input_tensor = ctx->Input(0);
    if (!ctx->ok()) {
      return;
    }

    std::vector<int64_t> out_dims = input_tensor.Shape();
    int64_t in_last_dim = out_dims.empty() ? 0 : out_dims.back();
    int64_t outer_dims =
        (in_last_dim > 0) ? (input_tensor.NumElements() / in_last_dim) : 0;
    if (!out_dims.empty()) {
      out_dims.back() = fft_length_;
    }

    TensorHandle output_tensor = ctx->AllocateOutput<T>(0, out_dims);
    if (!ctx->ok()) {
      return;
    }

    const T* input = input_tensor.Data<T>();
    T* output = output_tensor.Data<T>();
    for (int64_t i = 0; i < outer_dims; ++i) {
      ApplyFn(
          state_.data(),
          reinterpret_cast<const Complex<T>*>(&input[i * subband_array_size_]),
          &output[i * fft_length_]);
    }
  }

 private:
  int32_t fft_length_ = 0;
  int32_t subband_array_size_ = 0;
  std::vector<int8_t> state_;
};

class FftAutoScaleOp {
 public:
  explicit FftAutoScaleOp(KernelConstruction* /*ctx*/) {}

  void Compute(KernelContext* ctx) {
    TensorHandle input_tensor = ctx->Input(0);
    if (!ctx->ok()) {
      return;
    }

    TensorHandle output_tensor =
        ctx->AllocateOutput<int16_t>(0, input_tensor.Shape());
    TensorHandle scale_bit_tensor = ctx->AllocateScalarOutput<int32_t>(1);
    if (!ctx->ok()) {
      return;
    }

    *scale_bit_tensor.Data<int32_t>() = tflite::tflm_signal::FftAutoScale(
        input_tensor.Data<int16_t>(),
        static_cast<size_t>(input_tensor.NumElements()),
        output_tensor.Data<int16_t>());
  }
};

// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalFftKernels() {
  RegisterKernel<FftAutoScaleOp>("SignalFftAutoScale");

  RegisterTypedKernel<
      RfftOp<float, ::tflm_signal::RfftFloatGetNeededMemory,
             ::tflm_signal::RfftFloatInit, ::tflm_signal::RfftFloatApply>>(
      "SignalRfft", TF_FLOAT);
  RegisterTypedKernel<
      RfftOp<int16_t, ::tflm_signal::RfftInt16GetNeededMemory,
             ::tflm_signal::RfftInt16Init, ::tflm_signal::RfftInt16Apply>>(
      "SignalRfft", TF_INT16);
  RegisterTypedKernel<
      RfftOp<int32_t, ::tflm_signal::RfftInt32GetNeededMemory,
             ::tflm_signal::RfftInt32Init, ::tflm_signal::RfftInt32Apply>>(
      "SignalRfft", TF_INT32);

  RegisterTypedKernel<
      IrfftOp<float, tflite::tflm_signal::IrfftFloatGetNeededMemory,
              tflite::tflm_signal::IrfftFloatInit,
              tflite::tflm_signal::IrfftFloatApply>>("SignalIrfft", TF_FLOAT);
  RegisterTypedKernel<
      IrfftOp<int16_t, tflite::tflm_signal::IrfftInt16GetNeededMemory,
              tflite::tflm_signal::IrfftInt16Init,
              tflite::tflm_signal::IrfftInt16Apply>>("SignalIrfft", TF_INT16);
  RegisterTypedKernel<
      IrfftOp<int32_t, tflite::tflm_signal::IrfftInt32GetNeededMemory,
              tflite::tflm_signal::IrfftInt32Init,
              tflite::tflm_signal::IrfftInt32Apply>>("SignalIrfft", TF_INT32);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
