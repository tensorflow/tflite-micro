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
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace signal {
namespace {

template <typename T, TF_DataType E>
struct OverlapAddOp {
  int64_t frame_size = 0;
  int32_t frame_step = 0;
  int64_t n_frames = 0;
  int64_t outer_dims = 0;
  bool initialized = false;
  std::vector<std::vector<T>> state_buffers;

  static void* Create(TF_OpKernelConstruction* ctx) {
    auto* op = new OverlapAddOp{};
    TF_Status* status = TF_NewStatus();
    TF_OpKernelConstruction_GetAttrInt32(ctx, "frame_step", &op->frame_step,
                                         status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelConstruction_Failure(ctx, status);
      delete op;
      op = nullptr;
    }
    TF_DeleteStatus(status);
    return op;
  }

  static void Delete(void* kernel) {
    delete static_cast<OverlapAddOp*>(kernel);
  }

  static void Compute(void* kernel, TF_OpKernelContext* ctx) {
    auto* op = static_cast<OverlapAddOp*>(kernel);
    TF_Status* status = TF_NewStatus();
    TF_Tensor* input_tensor = nullptr;

    TF_GetInput(ctx, 0, &input_tensor, status);
    if (TF_GetCode(status) == TF_OK) {
      const int num_dims = TF_NumDims(input_tensor);
      if (!op->initialized && num_dims >= 2) {
        op->frame_size = TF_Dim(input_tensor, num_dims - 1);
        op->n_frames = TF_Dim(input_tensor, num_dims - 2);
        int64_t inner = op->n_frames * op->frame_size;
        op->outer_dims =
            (inner > 0) ? (TF_TensorElementCount(input_tensor) / inner) : 0;

        op->state_buffers.assign(
            static_cast<size_t>(op->outer_dims),
            std::vector<T>(static_cast<size_t>(op->frame_size), T{0}));
        op->initialized = true;
      }

      const int out_num_dims = num_dims - 1;
      std::vector<int64_t> out_dims(out_num_dims);
      for (int i = 0; i < out_num_dims - 1; ++i) {
        out_dims[i] = TF_Dim(input_tensor, i);
      }
      if (out_num_dims > 0) {
        out_dims[out_num_dims - 1] = op->n_frames * op->frame_step;
      }

      int64_t out_elements = op->outer_dims * op->n_frames * op->frame_step;
      TF_Tensor* output_tensor = TF_AllocateOutput(
          ctx, 0, E, out_dims.data(), out_num_dims,
          static_cast<size_t>(out_elements) * sizeof(T), status);
      if (TF_GetCode(status) == TF_OK) {
        const auto* input = static_cast<const T*>(TF_TensorData(input_tensor));
        auto* output = static_cast<T*>(TF_TensorData(output_tensor));

        for (int64_t i = 0; i < op->outer_dims; i++) {
          T* buffer = op->state_buffers[static_cast<size_t>(i)].data();
          for (int64_t frame = 0; frame < op->n_frames; frame++) {
            int64_t input_index = (i * op->n_frames + frame) * op->frame_size;
            int64_t output_index = (i * op->n_frames + frame) * op->frame_step;
            tflm_signal::OverlapAdd(&input[input_index], buffer,
                                    static_cast<int>(op->frame_size),
                                    &output[output_index], op->frame_step);
          }
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

template <typename Kernel>
void RegisterTypedOverlapAddKernel(TF_DataType type) {
  TF_Status* status = TF_NewStatus();
  auto* builder =
      TF_NewKernelBuilder("SignalOverlapAdd", "CPU", &Kernel::Create,
                          &Kernel::Compute, &Kernel::Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", type, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder("SignalOverlapAdd", builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  TF_DeleteStatus(status);
}

// TODO(b/286250473): change back name after name clash resolved
void RegisterSignalOverlapAddKernels() {
  RegisterTypedOverlapAddKernel<OverlapAddOp<float, TF_FLOAT>>(TF_FLOAT);
  RegisterTypedOverlapAddKernel<OverlapAddOp<int16_t, TF_INT16>>(TF_INT16);
}

const bool kSignalOverlapAddKernelsRegistered = []() {
  RegisterSignalOverlapAddKernels();
  return true;
}();

}  // namespace
}  // namespace signal
}  // namespace tensorflow
