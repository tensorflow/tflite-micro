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
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace signal {
namespace {

struct DelayOp {
  bool initialized = false;
  int64_t frame_size = 0;
  int32_t delay_length = 0;
  int64_t outer_dims = 0;
  std::vector<std::vector<int8_t>> state_buffers;
  std::vector<struct tflite::tflm_signal::CircularBuffer*> circular_buffers;
};

void* DelayOpCreate(TF_OpKernelConstruction* ctx) {
  auto* op = new DelayOp{};
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt32(ctx, "delay_length", &op->delay_length,
                                       status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    delete op;
    op = nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void DelayOpDelete(void* kernel) { delete static_cast<DelayOp*>(kernel); }

void DelayOpCompute(void* kernel, TF_OpKernelContext* ctx) {
  auto* op = static_cast<DelayOp*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;

  TF_GetInput(ctx, 0, &input_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    const int num_dims = TF_NumDims(input_tensor);
    std::vector<int64_t> dims(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      dims[i] = TF_Dim(input_tensor, i);
    }

    const int64_t total_elements = TF_TensorElementCount(input_tensor);
    if (!op->initialized) {
      op->frame_size = (num_dims > 0) ? dims[num_dims - 1] : 1;
      op->outer_dims =
          (op->frame_size > 0) ? (total_elements / op->frame_size) : 0;

      op->state_buffers.resize(static_cast<size_t>(op->outer_dims));
      op->circular_buffers.resize(static_cast<size_t>(op->outer_dims));

      // Calculate the capacity of the circular buffer.
      size_t capacity = static_cast<size_t>(op->frame_size + op->delay_length);
      size_t state_size =
          tflite::tflm_signal::CircularBufferGetNeededMemory(capacity);
      for (int64_t i = 0; i < op->outer_dims; i++) {
        op->state_buffers[static_cast<size_t>(i)].resize(state_size);
        int8_t* state = op->state_buffers[static_cast<size_t>(i)].data();
        op->circular_buffers[static_cast<size_t>(i)] =
            tflite::tflm_signal::CircularBufferInit(capacity, state,
                                                    state_size);
        tflite::tflm_signal::CircularBufferWriteZeros(
            op->circular_buffers[static_cast<size_t>(i)], op->delay_length);
      }
      op->initialized = true;
    }

    TF_Tensor* output_tensor = TF_AllocateOutput(
        ctx, 0, TF_INT16, dims.data(), num_dims,
        static_cast<size_t>(total_elements) * sizeof(int16_t), status);
    if (TF_GetCode(status) == TF_OK) {
      const auto* input =
          static_cast<const int16_t*>(TF_TensorData(input_tensor));
      auto* output = static_cast<int16_t*>(TF_TensorData(output_tensor));

      for (int64_t dim_index = 0, sample_index = 0; dim_index < op->outer_dims;
           dim_index++, sample_index += op->frame_size) {
        auto* cb = op->circular_buffers[static_cast<size_t>(dim_index)];
        tflite::tflm_signal::CircularBufferWrite(
            cb, &input[sample_index], static_cast<size_t>(op->frame_size));
        tflite::tflm_signal::CircularBufferGet(
            cb, static_cast<size_t>(op->frame_size), &output[sample_index]);
        tflite::tflm_signal::CircularBufferDiscard(
            cb, static_cast<size_t>(op->frame_size));
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

// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalDelayKernel() {
  TF_Status* status = TF_NewStatus();
  auto* builder = TF_NewKernelBuilder("SignalDelay", "CPU", &DelayOpCreate,
                                      &DelayOpCompute, &DelayOpDelete);
  TF_RegisterKernelBuilder("SignalDelay", builder, status);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
