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

// TODO(b/286250473): change back name after name clash resolved
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

}  // namespace signal
}  // namespace tensorflow