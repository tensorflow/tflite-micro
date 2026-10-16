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
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace signal {
namespace {

struct FilterBankOp {
  tflite::tflm_signal::FilterbankConfig config{};
  std::vector<uint64_t> work_area;
};

void* FilterBankOpCreate(TF_OpKernelConstruction* ctx) {
  auto* op = new FilterBankOp{};
  TF_Status* status = TF_NewStatus();
  int32_t num_channels = 0;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_channels", &num_channels,
                                       status);
  if (TF_GetCode(status) == TF_OK) {
    op->config.num_channels = num_channels;
    op->work_area.resize(static_cast<size_t>(num_channels) + 1);
  } else {
    TF_OpKernelConstruction_Failure(ctx, status);
    delete op;
    op = nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void FilterBankOpDelete(void* kernel) {
  delete static_cast<FilterBankOp*>(kernel);
}

void FilterBankOpCompute(void* kernel, TF_OpKernelContext* ctx) {
  auto* op = static_cast<FilterBankOp*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* in_tensors[6] = {nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr};

  for (int i = 0; i < 6 && TF_GetCode(status) == TF_OK; ++i) {
    TF_GetInput(ctx, i, &in_tensors[i], status);
  }

  if (TF_GetCode(status) == TF_OK) {
    const auto* input =
        static_cast<const uint32_t*>(TF_TensorData(in_tensors[0]));
    op->config.weights =
        static_cast<const int16_t*>(TF_TensorData(in_tensors[1]));
    op->config.unweights =
        static_cast<const int16_t*>(TF_TensorData(in_tensors[2]));
    op->config.channel_frequency_starts =
        static_cast<const int16_t*>(TF_TensorData(in_tensors[3]));
    op->config.channel_weight_starts =
        static_cast<const int16_t*>(TF_TensorData(in_tensors[4]));
    op->config.channel_widths =
        static_cast<const int16_t*>(TF_TensorData(in_tensors[5]));

    int64_t out_dims[1] = {op->config.num_channels};
    TF_Tensor* output_tensor = TF_AllocateOutput(
        ctx, 0, TF_UINT64, out_dims, 1,
        static_cast<size_t>(op->config.num_channels) * sizeof(uint64_t),
        status);
    if (TF_GetCode(status) == TF_OK) {
      tflite::tflm_signal::FilterbankAccumulateChannels(&op->config, input,
                                                        op->work_area.data());
      auto* output = static_cast<uint64_t*>(TF_TensorData(output_tensor));
      // Discard channel 0, which is just scratch.
      std::memcpy(
          output, op->work_area.data() + 1,
          sizeof(*output) * static_cast<size_t>(op->config.num_channels));
      TF_DeleteTensor(output_tensor);
    }
  }

  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
  }
  for (int i = 0; i < 6; ++i) {
    if (in_tensors[i] != nullptr) {
      TF_DeleteTensor(in_tensors[i]);
    }
  }
  TF_DeleteStatus(status);
}

void* FilterBankSquareRootOpCreate(TF_OpKernelConstruction* /*ctx*/) {
  return nullptr;
}
void FilterBankSquareRootOpDelete(void* /*kernel*/) {}

void FilterBankSquareRootOpCompute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;
  TF_Tensor* scale_bits_tensor = nullptr;

  TF_GetInput(ctx, 0, &input_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_GetInput(ctx, 1, &scale_bits_tensor, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    int64_t num_channels = TF_TensorElementCount(input_tensor);
    int64_t out_dims[1] = {num_channels};
    TF_Tensor* output_tensor = TF_AllocateOutput(
        ctx, 0, TF_UINT32, out_dims, 1,
        static_cast<size_t>(num_channels) * sizeof(uint32_t), status);
    if (TF_GetCode(status) == TF_OK) {
      const auto* input =
          static_cast<const uint64_t*>(TF_TensorData(input_tensor));
      int32_t scale_bits =
          *static_cast<const int32_t*>(TF_TensorData(scale_bits_tensor));
      auto* output = static_cast<uint32_t*>(TF_TensorData(output_tensor));
      tflite::tflm_signal::FilterbankSqrt(
          input, static_cast<int32_t>(num_channels), scale_bits, output);
      TF_DeleteTensor(output_tensor);
    }
  }

  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
  }
  if (input_tensor != nullptr) {
    TF_DeleteTensor(input_tensor);
  }
  if (scale_bits_tensor != nullptr) {
    TF_DeleteTensor(scale_bits_tensor);
  }
  TF_DeleteStatus(status);
}

struct FilterBankSpectralSubtractionOp {
  tflite::tflm_signal::SpectralSubtractionConfig config{};
  std::vector<uint32_t> noise_estimate;
};

void* FilterBankSpectralSubtractionOpCreate(TF_OpKernelConstruction* ctx) {
  auto* op = new FilterBankSpectralSubtractionOp{};
  TF_Status* status = TF_NewStatus();
  int32_t attr_int = 0;
  TF_Bool attr_bool = 0;

  auto get_int = [&](const char* name, auto* field) {
    if (TF_GetCode(status) == TF_OK) {
      TF_OpKernelConstruction_GetAttrInt32(ctx, name, &attr_int, status);
      if (TF_GetCode(status) == TF_OK) {
        *field = attr_int;
      }
    }
  };

  get_int("smoothing", &op->config.smoothing);
  get_int("one_minus_smoothing", &op->config.one_minus_smoothing);
  get_int("alternate_smoothing", &op->config.alternate_smoothing);
  get_int("alternate_one_minus_smoothing",
          &op->config.alternate_one_minus_smoothing);
  get_int("smoothing_bits", &op->config.smoothing_bits);
  get_int("min_signal_remaining", &op->config.min_signal_remaining);
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrBool(ctx, "clamping", &attr_bool, status);
    if (TF_GetCode(status) == TF_OK) {
      op->config.clamping = (attr_bool != 0);
    }
  }
  get_int("num_channels", &op->config.num_channels);
  get_int("spectral_subtraction_bits", &op->config.spectral_subtraction_bits);

  if (TF_GetCode(status) == TF_OK) {
    op->noise_estimate.assign(static_cast<size_t>(op->config.num_channels), 0);
  } else {
    TF_OpKernelConstruction_Failure(ctx, status);
    delete op;
    op = nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void FilterBankSpectralSubtractionOpDelete(void* kernel) {
  delete static_cast<FilterBankSpectralSubtractionOp*>(kernel);
}

void FilterBankSpectralSubtractionOpCompute(void* kernel,
                                            TF_OpKernelContext* ctx) {
  auto* op = static_cast<FilterBankSpectralSubtractionOp*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;

  TF_GetInput(ctx, 0, &input_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    int64_t out_dims[1] = {op->config.num_channels};
    size_t byte_len =
        static_cast<size_t>(op->config.num_channels) * sizeof(uint32_t);
    TF_Tensor* output_tensor =
        TF_AllocateOutput(ctx, 0, TF_UINT32, out_dims, 1, byte_len, status);
    TF_Tensor* noise_tensor = nullptr;
    if (TF_GetCode(status) == TF_OK) {
      noise_tensor =
          TF_AllocateOutput(ctx, 1, TF_UINT32, out_dims, 1, byte_len, status);
    }
    if (TF_GetCode(status) == TF_OK) {
      const auto* input =
          static_cast<const uint32_t*>(TF_TensorData(input_tensor));
      auto* output = static_cast<uint32_t*>(TF_TensorData(output_tensor));
      auto* noise_estimate =
          static_cast<uint32_t*>(TF_TensorData(noise_tensor));

      tflite::tflm_signal::FilterbankSpectralSubtraction(
          &op->config, input, output, op->noise_estimate.data());
      std::memcpy(noise_estimate, op->noise_estimate.data(), byte_len);
    }
    if (output_tensor != nullptr) {
      TF_DeleteTensor(output_tensor);
    }
    if (noise_tensor != nullptr) {
      TF_DeleteTensor(noise_tensor);
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

struct FilterBankLogOp {
  int32_t output_scale = 0;
  int32_t input_correction_bits = 0;
};

void* FilterBankLogOpCreate(TF_OpKernelConstruction* ctx) {
  auto* op = new FilterBankLogOp{};
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt32(ctx, "output_scale", &op->output_scale,
                                       status);
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrInt32(ctx, "input_correction_bits",
                                         &op->input_correction_bits, status);
  }
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    delete op;
    op = nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void FilterBankLogOpDelete(void* kernel) {
  delete static_cast<FilterBankLogOp*>(kernel);
}

void FilterBankLogOpCompute(void* kernel, TF_OpKernelContext* ctx) {
  auto* op = static_cast<FilterBankLogOp*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;

  TF_GetInput(ctx, 0, &input_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    int64_t num_channels = TF_TensorElementCount(input_tensor);
    int64_t out_dims[1] = {num_channels};
    TF_Tensor* output_tensor = TF_AllocateOutput(
        ctx, 0, TF_INT16, out_dims, 1,
        static_cast<size_t>(num_channels) * sizeof(int16_t), status);
    if (TF_GetCode(status) == TF_OK) {
      const auto* input =
          static_cast<const uint32_t*>(TF_TensorData(input_tensor));
      auto* output = static_cast<int16_t*>(TF_TensorData(output_tensor));
      tflite::tflm_signal::FilterbankLog(
          input, static_cast<int32_t>(num_channels), op->output_scale,
          op->input_correction_bits, output);
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

// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalFilterBankKernels() {
  TF_Status* status = TF_NewStatus();

  auto* fb_builder =
      TF_NewKernelBuilder("SignalFilterBank", "CPU", &FilterBankOpCreate,
                          &FilterBankOpCompute, &FilterBankOpDelete);
  TF_RegisterKernelBuilder("SignalFilterBank", fb_builder, status);

  auto* sqrt_builder = TF_NewKernelBuilder(
      "SignalFilterBankSquareRoot", "CPU", &FilterBankSquareRootOpCreate,
      &FilterBankSquareRootOpCompute, &FilterBankSquareRootOpDelete);
  TF_RegisterKernelBuilder("SignalFilterBankSquareRoot", sqrt_builder, status);

  auto* sub_builder =
      TF_NewKernelBuilder("SignalFilterBankSpectralSubtraction", "CPU",
                          &FilterBankSpectralSubtractionOpCreate,
                          &FilterBankSpectralSubtractionOpCompute,
                          &FilterBankSpectralSubtractionOpDelete);
  TF_RegisterKernelBuilder("SignalFilterBankSpectralSubtraction", sub_builder,
                           status);

  auto* log_builder =
      TF_NewKernelBuilder("SignalFilterBankLog", "CPU", &FilterBankLogOpCreate,
                          &FilterBankLogOpCompute, &FilterBankLogOpDelete);
  TF_RegisterKernelBuilder("SignalFilterBankLog", log_builder, status);

  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
