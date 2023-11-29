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

#include "signal/src/fft_auto_scale.h"
#include "signal/src/irfft.h"
#include "signal/src/rfft.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace signal {

// get_needed_memory_func(), init_func(), apply_func()
// are type specific implementations of the RFFT functions.
// See rfft.h included above for documentation
template <typename T, DataType E, size_t (*get_needed_memory_func)(int32_t),
          void* (*init_func)(int32_t, void*, size_t),
          void (*apply_func)(void*, const T* input, Complex<T>*)>
class RfftOp : public tensorflow::OpKernel {
 public:
  explicit RfftOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fft_length", &fft_length_));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(E, TensorShape({fft_length_}),
                                          &work_area_tensor_));
    work_area_ = work_area_tensor_.flat<T>().data();
    // Subband array size is the number of subbands * 2 because each coefficient
    // is complex.
    subband_array_size_ = ((fft_length_ / 2) + 1) * 2;
    size_t state_size = (*get_needed_memory_func)(fft_length_);
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DT_INT8, TensorShape({static_cast<int32_t>(state_size)}),
                       &state_tensor_));
    state_ = state_tensor_.flat<int8_t>().data();
    (*init_func)(fft_length_, state_, state_size);
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input_tensor = context->input(0);
    const T* input = input_tensor.flat<T>().data();

    TensorShape output_shape = input_tensor.shape();
    output_shape.set_dim(output_shape.dims() - 1, subband_array_size_);

    // Create an output tensor
    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    T* output = output_tensor->flat<T>().data();

    int outer_dims = input_tensor.flat_inner_dims<T, 2>().dimensions().at(0);
    int frame_size = input_tensor.flat_inner_dims<T, 2>().dimensions().at(1);
    for (int i = 0; i < outer_dims; i++) {
      auto input_in_work_end =
          std::copy_n(&input[i * frame_size], frame_size, work_area_);
      // Zero pad input to FFT length
      std::fill(input_in_work_end, &work_area_[fft_length_], 0);
      (*apply_func)(
          state_, work_area_,
          reinterpret_cast<Complex<T>*>(&output[i * subband_array_size_]));
    }
  }

 private:
  int fft_length_;
  int subband_array_size_;
  int8_t* state_;
  T* work_area_;
  Tensor work_area_tensor_;
  Tensor state_tensor_;
};

// get_needed_memory_func(), init_func(), apply_func()
// are type specific implementations of the IRFFT functions.
// See irfft.h included above for documentation
template <typename T, size_t (*get_needed_memory_func)(int32_t),
          void* (*init_func)(int32_t, void*, size_t),
          void (*apply_func)(void*, const Complex<T>* input, T*)>
class IrfftOp : public tensorflow::OpKernel {
 public:
  explicit IrfftOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fft_length", &fft_length_));
    // Subband array size is the number of subbands * 2 because each coefficient
    // is complex.
    subband_array_size_ = ((fft_length_ / 2) + 1) * 2;

    size_t state_size = (*get_needed_memory_func)(fft_length_);
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT8, TensorShape({(int32_t)state_size}),
                                &state_handle_));
    state_ = state_handle_.flat<int8_t>().data();
    (*init_func)(fft_length_, state_, state_size);
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input_tensor = context->input(0);
    const T* input = input_tensor.flat<T>().data();

    TensorShape output_shape = input_tensor.shape();
    output_shape.set_dim(output_shape.dims() - 1, fft_length_);

    // Create an output tensor
    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    T* output = output_tensor->flat<T>().data();

    int outer_dims = input_tensor.flat_inner_dims<T, 2>().dimensions().at(0);
    for (int i = 0; i < outer_dims; i++) {
      (*apply_func)(
          state_,
          reinterpret_cast<const Complex<T>*>(&input[i * subband_array_size_]),
          &output[i * fft_length_]);
    }
  }

 private:
  int fft_length_;
  int subband_array_size_;
  int8_t* state_;
  Tensor state_handle_;
};

class FftAutoScaleOp : public tensorflow::OpKernel {
 public:
  explicit FftAutoScaleOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}
  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input_tensor = context->input(0);
    const int16_t* input = input_tensor.flat<int16_t>().data();

    // Create an output tensor
    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    int16_t* output = output_tensor->flat<int16_t>().data();

    tensorflow::Tensor* scale_bit_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &scale_bit_tensor));
    scale_bit_tensor->scalar<int32_t>()() = tflite::tflm_signal::FftAutoScale(
        input, output_tensor->NumElements(), output);
  }
};

// TODO(b/286250473): change back name after name clash resolved
REGISTER_KERNEL_BUILDER(
    Name("SignalFftAutoScale").Device(tensorflow::DEVICE_CPU), FftAutoScaleOp);
REGISTER_KERNEL_BUILDER(
    Name("SignalRfft")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<float>("T"),
    RfftOp<float, DT_FLOAT, ::tflm_signal::RfftFloatGetNeededMemory,
           ::tflm_signal::RfftFloatInit, ::tflm_signal::RfftFloatApply>);
REGISTER_KERNEL_BUILDER(
    Name("SignalRfft")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<int16>("T"),
    RfftOp<int16_t, DT_INT16, ::tflm_signal::RfftInt16GetNeededMemory,
           ::tflm_signal::RfftInt16Init, ::tflm_signal::RfftInt16Apply>);
REGISTER_KERNEL_BUILDER(
    Name("SignalRfft")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<int32>("T"),
    RfftOp<int32_t, DT_INT32, ::tflm_signal::RfftInt32GetNeededMemory,
           ::tflm_signal::RfftInt32Init, ::tflm_signal::RfftInt32Apply>);

REGISTER_KERNEL_BUILDER(
    Name("SignalIrfft")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<float>("T"),
    IrfftOp<float, tflite::tflm_signal::IrfftFloatGetNeededMemory,
            tflite::tflm_signal::IrfftFloatInit,
            tflite::tflm_signal::IrfftFloatApply>);
REGISTER_KERNEL_BUILDER(
    Name("SignalIrfft")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<int16>("T"),
    IrfftOp<int16_t, tflite::tflm_signal::IrfftInt16GetNeededMemory,
            tflite::tflm_signal::IrfftInt16Init,
            tflite::tflm_signal::IrfftInt16Apply>);
REGISTER_KERNEL_BUILDER(
    Name("SignalIrfft")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<int32>("T"),
    IrfftOp<int32_t, tflite::tflm_signal::IrfftInt32GetNeededMemory,
            tflite::tflm_signal::IrfftInt32Init,
            tflite::tflm_signal::IrfftInt32Apply>);

}  // namespace signal
}  // namespace tensorflow