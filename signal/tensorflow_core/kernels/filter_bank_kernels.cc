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
#include <cstring>
#include <vector>

#include "signal/src/filter_bank.h"
#include "signal/src/filter_bank_log.h"
#include "signal/src/filter_bank_spectral_subtraction.h"
#include "signal/src/filter_bank_square_root.h"
#include "signal/tensorflow_core/kernels/kernel_utils.h"

namespace tensorflow {
namespace signal {
namespace {

class FilterBankOp {
 public:
  explicit FilterBankOp(KernelConstruction* ctx) {
    int32_t num_channels = 0;
    if (ctx->GetAttr("num_channels", &num_channels)) {
      config_.num_channels = num_channels;
      work_area_.resize(static_cast<size_t>(num_channels) + 1);
    }
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input = ctx->Input(0);
    TensorHandle weights = ctx->Input(1);
    TensorHandle unweights = ctx->Input(2);
    TensorHandle freq_starts = ctx->Input(3);
    TensorHandle weight_starts = ctx->Input(4);
    TensorHandle widths = ctx->Input(5);
    TensorHandle output =
        ctx->AllocateOutput<uint64_t>(0, {config_.num_channels});
    if (!ctx->ok()) {
      return;
    }

    config_.weights = weights.Data<int16_t>();
    config_.unweights = unweights.Data<int16_t>();
    config_.channel_frequency_starts = freq_starts.Data<int16_t>();
    config_.channel_weight_starts = weight_starts.Data<int16_t>();
    config_.channel_widths = widths.Data<int16_t>();

    tflite::tflm_signal::FilterbankAccumulateChannels(
        &config_, input.Data<uint32_t>(), work_area_.data());

    // Discard channel 0, which is scratch space.
    std::memcpy(output.Data<uint64_t>(), work_area_.data() + 1,
                sizeof(uint64_t) * static_cast<size_t>(config_.num_channels));
  }

 private:
  tflite::tflm_signal::FilterbankConfig config_{};
  std::vector<uint64_t> work_area_;
};

class FilterBankSquareRootOp {
 public:
  explicit FilterBankSquareRootOp(KernelConstruction* /*ctx*/) {}

  void Compute(KernelContext* ctx) {
    TensorHandle input = ctx->Input(0);
    TensorHandle scale_bits = ctx->Input(1);
    if (!ctx->ok()) {
      return;
    }

    int64_t num_channels = input.NumElements();
    TensorHandle output = ctx->AllocateOutput<uint32_t>(0, {num_channels});
    if (!ctx->ok()) {
      return;
    }

    tflite::tflm_signal::FilterbankSqrt(
        input.Data<uint64_t>(), static_cast<int32_t>(num_channels),
        *scale_bits.Data<int32_t>(), output.Data<uint32_t>());
  }
};

class FilterBankSpectralSubtractionOp {
 public:
  explicit FilterBankSpectralSubtractionOp(KernelConstruction* ctx) {
    auto get_int = [ctx](const char* name, auto* field) {
      int32_t val = 0;
      if (ctx->GetAttr(name, &val)) {
        *field = val;
      }
    };

    get_int("smoothing", &config_.smoothing);
    get_int("one_minus_smoothing", &config_.one_minus_smoothing);
    get_int("alternate_smoothing", &config_.alternate_smoothing);
    get_int("alternate_one_minus_smoothing",
            &config_.alternate_one_minus_smoothing);
    get_int("smoothing_bits", &config_.smoothing_bits);
    get_int("min_signal_remaining", &config_.min_signal_remaining);
    bool clamping = false;
    if (ctx->GetAttr("clamping", &clamping)) {
      config_.clamping = clamping;
    }
    get_int("num_channels", &config_.num_channels);
    get_int("spectral_subtraction_bits", &config_.spectral_subtraction_bits);

    if (ctx->ok()) {
      noise_estimate_.assign(static_cast<size_t>(config_.num_channels), 0);
    }
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input = ctx->Input(0);
    TensorHandle output =
        ctx->AllocateOutput<uint32_t>(0, {config_.num_channels});
    TensorHandle noise_out =
        ctx->AllocateOutput<uint32_t>(1, {config_.num_channels});
    if (!ctx->ok()) {
      return;
    }

    tflite::tflm_signal::FilterbankSpectralSubtraction(
        &config_, input.Data<uint32_t>(), output.Data<uint32_t>(),
        noise_estimate_.data());
    std::memcpy(noise_out.Data<uint32_t>(), noise_estimate_.data(),
                sizeof(uint32_t) * static_cast<size_t>(config_.num_channels));
  }

 private:
  tflite::tflm_signal::SpectralSubtractionConfig config_{};
  std::vector<uint32_t> noise_estimate_;
};

class FilterBankLogOp {
 public:
  explicit FilterBankLogOp(KernelConstruction* ctx) {
    ctx->GetAttr("output_scale", &output_scale_);
    ctx->GetAttr("input_correction_bits", &input_correction_bits_);
  }

  void Compute(KernelContext* ctx) {
    TensorHandle input = ctx->Input(0);
    if (!ctx->ok()) {
      return;
    }

    int64_t num_channels = input.NumElements();
    TensorHandle output = ctx->AllocateOutput<int16_t>(0, {num_channels});
    if (!ctx->ok()) {
      return;
    }

    tflite::tflm_signal::FilterbankLog(
        input.Data<uint32_t>(), static_cast<int32_t>(num_channels),
        output_scale_, input_correction_bits_, output.Data<int16_t>());
  }

 private:
  int32_t output_scale_ = 0;
  int32_t input_correction_bits_ = 0;
};

// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalFilterBankKernels() {
  RegisterKernel<FilterBankOp>("SignalFilterBank");
  RegisterKernel<FilterBankSquareRootOp>("SignalFilterBankSquareRoot");
  RegisterKernel<FilterBankSpectralSubtractionOp>(
      "SignalFilterBankSpectralSubtraction");
  RegisterKernel<FilterBankLogOp>("SignalFilterBankLog");
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
