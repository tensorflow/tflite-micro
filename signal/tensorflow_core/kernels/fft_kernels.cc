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
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace signal {
namespace {

// get_needed_memory_func(), init_func(), apply_func()
// are type specific implementations of the RFFT functions.
// See rfft.h included above for documentation.
template <typename T, TF_DataType E, size_t (*get_needed_memory_func)(int32_t),
          void* (*init_func)(int32_t, void*, size_t),
          void (*apply_func)(void*, const T* input, Complex<T>*)>
struct RfftOp {
  int32_t fft_length = 0;
  int32_t subband_array_size = 0;
  std::vector<T> work_area;
  std::vector<int8_t> state;

  static void* Create(TF_OpKernelConstruction* ctx) {
    auto* op = new RfftOp{};
    TF_Status* status = TF_NewStatus();
    TF_OpKernelConstruction_GetAttrInt32(ctx, "fft_length", &op->fft_length,
                                         status);
    if (TF_GetCode(status) == TF_OK) {
      op->work_area.resize(static_cast<size_t>(op->fft_length));
      // Subband array size is the number of subbands * 2 because each
      // coefficient is complex.
      op->subband_array_size = ((op->fft_length / 2) + 1) * 2;
      size_t state_size = (*get_needed_memory_func)(op->fft_length);
      op->state.resize(state_size);
      (*init_func)(op->fft_length, op->state.data(), state_size);
    } else {
      TF_OpKernelConstruction_Failure(ctx, status);
      delete op;
      op = nullptr;
    }
    TF_DeleteStatus(status);
    return op;
  }

  static void Delete(void* kernel) { delete static_cast<RfftOp*>(kernel); }

  static void Compute(void* kernel, TF_OpKernelContext* ctx) {
    auto* op = static_cast<RfftOp*>(kernel);
    TF_Status* status = TF_NewStatus();
    TF_Tensor* input_tensor = nullptr;

    TF_GetInput(ctx, 0, &input_tensor, status);
    if (TF_GetCode(status) == TF_OK) {
      const int num_dims = TF_NumDims(input_tensor);
      std::vector<int64_t> out_dims(num_dims);
      for (int i = 0; i < num_dims; ++i) {
        out_dims[i] = TF_Dim(input_tensor, i);
      }
      int64_t frame_size = (num_dims > 0) ? out_dims[num_dims - 1] : 0;
      int64_t total_elements = TF_TensorElementCount(input_tensor);
      int64_t outer_dims = (frame_size > 0) ? (total_elements / frame_size) : 0;
      if (num_dims > 0) {
        out_dims[num_dims - 1] = op->subband_array_size;
      }

      TF_Tensor* output_tensor = TF_AllocateOutput(
          ctx, 0, E, out_dims.data(), num_dims,
          static_cast<size_t>(outer_dims * op->subband_array_size) * sizeof(T),
          status);
      if (TF_GetCode(status) == TF_OK) {
        const auto* input = static_cast<const T*>(TF_TensorData(input_tensor));
        auto* output = static_cast<T*>(TF_TensorData(output_tensor));

        for (int64_t i = 0; i < outer_dims; i++) {
          auto input_in_work_end = std::copy_n(
              &input[i * frame_size], frame_size, op->work_area.data());
          // Zero pad input to FFT length.
          std::fill(input_in_work_end, op->work_area.data() + op->fft_length,
                    T{0});
          (*apply_func)(op->state.data(), op->work_area.data(),
                        reinterpret_cast<Complex<T>*>(
                            &output[i * op->subband_array_size]));
        }
        TF_DeleteTensor(output_tensor);
      }
    }

    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
    }
    if (input_tensor != nullptr) {
      TF_DeleteTensor(input_tensor);
    }
    TF_DeleteStatus(status);
  }
};

// get_needed_memory_func(), init_func(), apply_func()
// are type specific implementations of the IRFFT functions.
// See irfft.h included above for documentation.
template <typename T, TF_DataType E, size_t (*get_needed_memory_func)(int32_t),
          void* (*init_func)(int32_t, void*, size_t),
          void (*apply_func)(void*, const Complex<T>* input, T*)>
struct IrfftOp {
  int32_t fft_length = 0;
  int32_t subband_array_size = 0;
  std::vector<int8_t> state;

  static void* Create(TF_OpKernelConstruction* ctx) {
    auto* op = new IrfftOp{};
    TF_Status* status = TF_NewStatus();
    TF_OpKernelConstruction_GetAttrInt32(ctx, "fft_length", &op->fft_length,
                                         status);
    if (TF_GetCode(status) == TF_OK) {
      // Subband array size is the number of subbands * 2 because each
      // coefficient is complex.
      op->subband_array_size = ((op->fft_length / 2) + 1) * 2;
      size_t state_size = (*get_needed_memory_func)(op->fft_length);
      op->state.resize(state_size);
      (*init_func)(op->fft_length, op->state.data(), state_size);
    } else {
      TF_OpKernelConstruction_Failure(ctx, status);
      delete op;
      op = nullptr;
    }
    TF_DeleteStatus(status);
    return op;
  }

  static void Delete(void* kernel) { delete static_cast<IrfftOp*>(kernel); }

  static void Compute(void* kernel, TF_OpKernelContext* ctx) {
    auto* op = static_cast<IrfftOp*>(kernel);
    TF_Status* status = TF_NewStatus();
    TF_Tensor* input_tensor = nullptr;

    TF_GetInput(ctx, 0, &input_tensor, status);
    if (TF_GetCode(status) == TF_OK) {
      const int num_dims = TF_NumDims(input_tensor);
      std::vector<int64_t> out_dims(num_dims);
      for (int i = 0; i < num_dims; ++i) {
        out_dims[i] = TF_Dim(input_tensor, i);
      }
      int64_t in_last_dim = (num_dims > 0) ? out_dims[num_dims - 1] : 0;
      int64_t total_elements = TF_TensorElementCount(input_tensor);
      int64_t outer_dims =
          (in_last_dim > 0) ? (total_elements / in_last_dim) : 0;
      if (num_dims > 0) {
        out_dims[num_dims - 1] = op->fft_length;
      }

      TF_Tensor* output_tensor = TF_AllocateOutput(
          ctx, 0, E, out_dims.data(), num_dims,
          static_cast<size_t>(outer_dims * op->fft_length) * sizeof(T), status);
      if (TF_GetCode(status) == TF_OK) {
        const auto* input = static_cast<const T*>(TF_TensorData(input_tensor));
        auto* output = static_cast<T*>(TF_TensorData(output_tensor));

        for (int64_t i = 0; i < outer_dims; i++) {
          (*apply_func)(op->state.data(),
                        reinterpret_cast<const Complex<T>*>(
                            &input[i * op->subband_array_size]),
                        &output[i * op->fft_length]);
        }
        TF_DeleteTensor(output_tensor);
      }
    }

    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
    }
    if (input_tensor != nullptr) {
      TF_DeleteTensor(input_tensor);
    }
    TF_DeleteStatus(status);
  }
};

void* FftAutoScaleOpCreate(TF_OpKernelConstruction* /*ctx*/) { return nullptr; }
void FftAutoScaleOpDelete(void* /*kernel*/) {}

void FftAutoScaleOpCompute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;

  TF_GetInput(ctx, 0, &input_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    const int num_dims = TF_NumDims(input_tensor);
    std::vector<int64_t> dims(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      dims[i] = TF_Dim(input_tensor, i);
    }
    int64_t total_elements = TF_TensorElementCount(input_tensor);

    TF_Tensor* output_tensor = TF_AllocateOutput(
        ctx, 0, TF_INT16, dims.data(), num_dims,
        static_cast<size_t>(total_elements) * sizeof(int16_t), status);
    TF_Tensor* scale_bit_tensor = nullptr;
    if (TF_GetCode(status) == TF_OK) {
      scale_bit_tensor = TF_AllocateOutput(ctx, 1, TF_INT32, nullptr, 0,
                                           sizeof(int32_t), status);
    }
    if (TF_GetCode(status) == TF_OK) {
      const auto* input =
          static_cast<const int16_t*>(TF_TensorData(input_tensor));
      auto* output = static_cast<int16_t*>(TF_TensorData(output_tensor));
      auto* scale_bit = static_cast<int32_t*>(TF_TensorData(scale_bit_tensor));
      *scale_bit = tflite::tflm_signal::FftAutoScale(
          input, static_cast<size_t>(total_elements), output);
    }
    if (output_tensor != nullptr) {
      TF_DeleteTensor(output_tensor);
    }
    if (scale_bit_tensor != nullptr) {
      TF_DeleteTensor(scale_bit_tensor);
    }
  }

  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
  }
  if (input_tensor != nullptr) {
    TF_DeleteTensor(input_tensor);
  }
  TF_DeleteStatus(status);
}

template <typename Kernel>
void RegisterTypedKernel(const char* op_name, TF_DataType type) {
  TF_Status* status = TF_NewStatus();
  auto* builder = TF_NewKernelBuilder(op_name, "CPU", &Kernel::Create,
                                      &Kernel::Compute, &Kernel::Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", type, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(op_name, builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  TF_DeleteStatus(status);
}

// TODO(b/286250473): change back name after name clash resolved
void RegisterSignalFftKernels() {
  TF_Status* status = TF_NewStatus();
  auto* auto_scale_builder =
      TF_NewKernelBuilder("SignalFftAutoScale", "CPU", &FftAutoScaleOpCreate,
                          &FftAutoScaleOpCompute, &FftAutoScaleOpDelete);
  TF_RegisterKernelBuilder("SignalFftAutoScale", auto_scale_builder, status);
  TF_DeleteStatus(status);

  RegisterTypedKernel<
      RfftOp<float, TF_FLOAT, ::tflm_signal::RfftFloatGetNeededMemory,
             ::tflm_signal::RfftFloatInit, ::tflm_signal::RfftFloatApply>>(
      "SignalRfft", TF_FLOAT);
  RegisterTypedKernel<
      RfftOp<int16_t, TF_INT16, ::tflm_signal::RfftInt16GetNeededMemory,
             ::tflm_signal::RfftInt16Init, ::tflm_signal::RfftInt16Apply>>(
      "SignalRfft", TF_INT16);
  RegisterTypedKernel<
      RfftOp<int32_t, TF_INT32, ::tflm_signal::RfftInt32GetNeededMemory,
             ::tflm_signal::RfftInt32Init, ::tflm_signal::RfftInt32Apply>>(
      "SignalRfft", TF_INT32);

  RegisterTypedKernel<
      IrfftOp<float, TF_FLOAT, tflite::tflm_signal::IrfftFloatGetNeededMemory,
              tflite::tflm_signal::IrfftFloatInit,
              tflite::tflm_signal::IrfftFloatApply>>("SignalIrfft", TF_FLOAT);
  RegisterTypedKernel<
      IrfftOp<int16_t, TF_INT16, tflite::tflm_signal::IrfftInt16GetNeededMemory,
              tflite::tflm_signal::IrfftInt16Init,
              tflite::tflm_signal::IrfftInt16Apply>>("SignalIrfft", TF_INT16);
  RegisterTypedKernel<
      IrfftOp<int32_t, TF_INT32, tflite::tflm_signal::IrfftInt32GetNeededMemory,
              tflite::tflm_signal::IrfftInt32Init,
              tflite::tflm_signal::IrfftInt32Apply>>("SignalIrfft", TF_INT32);
}

const bool kSignalFftKernelsRegistered = []() {
  RegisterSignalFftKernels();
  return true;
}();

}  // namespace
}  // namespace signal
}  // namespace tensorflow
