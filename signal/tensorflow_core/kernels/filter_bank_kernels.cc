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

#include "signal/src/filter_bank.h"
#include "signal/src/filter_bank_log.h"
#include "signal/src/filter_bank_spectral_subtraction.h"
#include "signal/src/filter_bank_square_root.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace signal {

class FilterBankOp : public tensorflow::OpKernel {
 public:
  explicit FilterBankOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    int32_t num_channels;
    OP_REQUIRES_OK(context, context->GetAttr("num_channels", &num_channels));
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_UINT64, TensorShape({num_channels + 1}),
                                &work_area_tensor_));
    work_area_ = work_area_tensor_.flat<uint64_t>().data();
    config_.num_channels = num_channels;
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input_tensor = context->input(0);
    const uint32_t* input = input_tensor.flat<uint32_t>().data();

    config_.weights = context->input(1).flat<int16_t>().data();
    config_.unweights = context->input(2).flat<int16_t>().data();
    config_.channel_frequency_starts = context->input(3).flat<int16_t>().data();
    config_.channel_weight_starts = context->input(4).flat<int16_t>().data();
    config_.channel_widths = context->input(5).flat<int16_t>().data();

    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {config_.num_channels},
                                                     &output_tensor));
    tflite::tflm_signal::FilterbankAccumulateChannels(&config_, input,
                                                      work_area_);

    uint64_t* output = output_tensor->flat<uint64_t>().data();
    // Discard channel 0, which is just scratch
    memcpy(output, work_area_ + 1, sizeof(*output) * config_.num_channels);
  }

 private:
  tflite::tflm_signal::FilterbankConfig config_;
  uint64_t* work_area_;
  Tensor work_area_tensor_;
};

class FilterBankSquareRootOp : public tensorflow::OpKernel {
 public:
  explicit FilterBankSquareRootOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const uint64_t* input = context->input(0).flat<uint64_t>().data();
    int32_t scale_bits = context->input(1).scalar<int32_t>()();
    int32_t num_channels = context->input(0).NumElements();

    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {num_channels}, &output_tensor));
    uint32_t* output = output_tensor->flat<uint32_t>().data();
    tflite::tflm_signal::FilterbankSqrt(input, num_channels, scale_bits,
                                        output);
  }

 private:
};

class FilterBankSpectralSubtractionOp : public tensorflow::OpKernel {
 public:
  explicit FilterBankSpectralSubtractionOp(
      tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    int attr_int;
    bool attr_bool;
    OP_REQUIRES_OK(context, context->GetAttr("smoothing", &attr_int));
    config_.smoothing = attr_int;
    OP_REQUIRES_OK(context, context->GetAttr("one_minus_smoothing", &attr_int));
    config_.one_minus_smoothing = attr_int;
    OP_REQUIRES_OK(context, context->GetAttr("alternate_smoothing", &attr_int));
    config_.alternate_smoothing = attr_int;
    OP_REQUIRES_OK(
        context, context->GetAttr("alternate_one_minus_smoothing", &attr_int));
    config_.alternate_one_minus_smoothing = attr_int;
    OP_REQUIRES_OK(context, context->GetAttr("smoothing_bits", &attr_int));
    config_.smoothing_bits = attr_int;
    OP_REQUIRES_OK(context,
                   context->GetAttr("min_signal_remaining", &attr_int));
    config_.min_signal_remaining = attr_int;
    OP_REQUIRES_OK(context, context->GetAttr("clamping", &attr_bool));
    config_.clamping = attr_bool;
    OP_REQUIRES_OK(context, context->GetAttr("num_channels", &attr_int));
    config_.num_channels = attr_int;
    OP_REQUIRES_OK(context,
                   context->GetAttr("spectral_subtraction_bits", &attr_int));
    config_.spectral_subtraction_bits = attr_int;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_UINT32, TensorShape({config_.num_channels}),
                                &noise_estimate_tensor_));
    noise_estimate_ = (uint32_t*)noise_estimate_tensor_.flat<uint32_t>().data();
    memset(noise_estimate_, 0, sizeof(uint32_t) * config_.num_channels);
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    tensorflow::Tensor* output_tensor = nullptr;
    const uint32_t* input = context->input(0).flat<uint32_t>().data();
    OP_REQUIRES_OK(context, context->allocate_output(0, {config_.num_channels},
                                                     &output_tensor));
    uint32_t* output = output_tensor->flat<uint32_t>().data();
    OP_REQUIRES_OK(context, context->allocate_output(1, {config_.num_channels},
                                                     &output_tensor));
    uint32_t* noise_estimate = output_tensor->flat<uint32_t>().data();

    tflite::tflm_signal::FilterbankSpectralSubtraction(&config_, input, output,
                                                       noise_estimate_);
    memcpy(noise_estimate, noise_estimate_,
           sizeof(*noise_estimate) * config_.num_channels);
  }

 private:
  Tensor noise_estimate_tensor_;
  tflite::tflm_signal::SpectralSubtractionConfig config_;
  uint32_t* noise_estimate_;
};

class FilterBankLogOp : public tensorflow::OpKernel {
 public:
  explicit FilterBankLogOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("output_scale", &output_scale_));
    OP_REQUIRES_OK(context, context->GetAttr("input_correction_bits",
                                             &input_correction_bits_));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const uint32_t* input = context->input(0).flat<uint32_t>().data();
    int num_channels = context->input(0).NumElements();
    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {num_channels}, &output_tensor));
    int16_t* output = output_tensor->flat<int16_t>().data();
    tflite::tflm_signal::FilterbankLog(input, num_channels, output_scale_,
                                       input_correction_bits_, output);
  }

 private:
  int output_scale_;
  int input_correction_bits_;
};

// TODO(b/286250473): change back name after name clash resolved
REGISTER_KERNEL_BUILDER(Name("SignalFilterBank").Device(tensorflow::DEVICE_CPU),
                        FilterBankOp);
REGISTER_KERNEL_BUILDER(
    Name("SignalFilterBankSquareRoot").Device(tensorflow::DEVICE_CPU),
    FilterBankSquareRootOp);
REGISTER_KERNEL_BUILDER(
    Name("SignalFilterBankSpectralSubtraction").Device(tensorflow::DEVICE_CPU),
    FilterBankSpectralSubtractionOp);
REGISTER_KERNEL_BUILDER(
    Name("SignalFilterBankLog").Device(tensorflow::DEVICE_CPU),
    FilterBankLogOp);

}  // namespace signal
}  // namespace tensorflow
