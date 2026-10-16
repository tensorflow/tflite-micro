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

struct FramerOp {
  bool initialized = false;
  int32_t frame_size = 0;
  int32_t frame_step = 0;
  int64_t outer_dims = 0;
  bool prefill = false;
  int64_t n_frames = 0;
  std::vector<std::vector<int8_t>> state_buffers;
  std::vector<struct tflite::tflm_signal::CircularBuffer*> circular_buffers;
};

void* FramerOpCreate(TF_OpKernelConstruction* ctx) {
  auto* op = new FramerOp{};
  TF_Status* status = TF_NewStatus();
  TF_Bool prefill_bool = 0;

  TF_OpKernelConstruction_GetAttrInt32(ctx, "frame_size", &op->frame_size,
                                       status);
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrInt32(ctx, "frame_step", &op->frame_step,
                                         status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrBool(ctx, "prefill", &prefill_bool, status);
    if (TF_GetCode(status) == TF_OK) {
      op->prefill = (prefill_bool != 0);
    }
  }
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    delete op;
    op = nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void FramerOpDelete(void* kernel) { delete static_cast<FramerOp*>(kernel); }

void FramerOpCompute(void* kernel, TF_OpKernelContext* ctx) {
  auto* op = static_cast<FramerOp*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;

  TF_GetInput(ctx, 0, &input_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    const int num_dims = TF_NumDims(input_tensor);
    const int64_t last_dim =
        (num_dims > 0) ? TF_Dim(input_tensor, num_dims - 1) : 0;
    const int64_t total_elements = TF_TensorElementCount(input_tensor);

    if (!op->initialized) {
      op->n_frames = last_dim / op->frame_step;
      op->outer_dims = (last_dim > 0) ? (total_elements / last_dim) : 0;

      op->state_buffers.resize(static_cast<size_t>(op->outer_dims));
      op->circular_buffers.resize(static_cast<size_t>(op->outer_dims));

      // Calculate the capacity of the circular buffer. Round up the frame size
      // to a multiple of frame step. Saves memory relative to the simpler
      // frame_size + frame_step. For example:
      // step_size = 160, frame_size = 400
      // capacity = 480 vs. step_size + frame_size = 560
      size_t capacity =
          static_cast<size_t>((op->frame_size + op->frame_step - 1) /
                              op->frame_step) *
          op->frame_step;
      size_t state_size =
          tflite::tflm_signal::CircularBufferGetNeededMemory(capacity);
      for (int64_t i = 0; i < op->outer_dims; i++) {
        op->state_buffers[static_cast<size_t>(i)].resize(state_size);
        int8_t* state = op->state_buffers[static_cast<size_t>(i)].data();
        op->circular_buffers[static_cast<size_t>(i)] =
            tflite::tflm_signal::CircularBufferInit(capacity, state,
                                                    state_size);
        if (op->prefill) {
          tflite::tflm_signal::CircularBufferWriteZeros(
              op->circular_buffers[static_cast<size_t>(i)],
              op->frame_size - op->frame_step);
        }
      }
      op->initialized = true;
    }

    // Split the last dimension of the input into {n_frames, frame_size}.
    std::vector<int64_t> out_dims(num_dims + 1);
    for (int i = 0; i < num_dims - 1; ++i) {
      out_dims[i] = TF_Dim(input_tensor, i);
    }
    if (num_dims > 0) {
      out_dims[num_dims - 1] = op->n_frames;
      out_dims[num_dims] = op->frame_size;
    }

    int64_t out_elements = op->outer_dims * op->n_frames * op->frame_size;
    TF_Tensor* output_tensor = TF_AllocateOutput(
        ctx, 0, TF_INT16, out_dims.data(), num_dims + 1,
        static_cast<size_t>(out_elements) * sizeof(int16_t), status);
    TF_Tensor* output_valid_tensor = nullptr;
    if (TF_GetCode(status) == TF_OK) {
      output_valid_tensor =
          TF_AllocateOutput(ctx, 1, TF_BOOL, nullptr, 0, sizeof(bool), status);
    }
    if (TF_GetCode(status) == TF_OK) {
      const auto* input =
          static_cast<const int16_t*>(TF_TensorData(input_tensor));
      auto* output = static_cast<int16_t*>(TF_TensorData(output_tensor));
      auto* out_valid = static_cast<bool*>(TF_TensorData(output_valid_tensor));

      bool output_valid = true;
      for (int64_t i = 0; i < op->outer_dims; i++) {
        auto* cb = op->circular_buffers[static_cast<size_t>(i)];
        for (int64_t frame = 0; frame < op->n_frames; frame++) {
          int64_t input_index = (i * op->n_frames + frame) * op->frame_step;
          int64_t output_index = (i * op->n_frames + frame) * op->frame_size;
          tflite::tflm_signal::CircularBufferWrite(
              cb, &input[input_index], static_cast<size_t>(op->frame_step));
          if (tflite::tflm_signal::CircularBufferAvailable(cb) >=
              static_cast<size_t>(op->frame_size)) {
            tflite::tflm_signal::CircularBufferGet(
                cb, static_cast<size_t>(op->frame_size), &output[output_index]);
            tflite::tflm_signal::CircularBufferDiscard(
                cb, static_cast<size_t>(op->frame_step));
          } else {
            output_valid = false;
          }
        }
        *out_valid = output_valid;
      }
    }
    if (output_tensor != nullptr) {
      TF_DeleteTensor(output_tensor);
    }
    if (output_valid_tensor != nullptr) {
      TF_DeleteTensor(output_valid_tensor);
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
__attribute__((constructor)) void RegisterSignalFramerKernel() {
  TF_Status* status = TF_NewStatus();
  auto* builder = TF_NewKernelBuilder("SignalFramer", "CPU", &FramerOpCreate,
                                      &FramerOpCompute, &FramerOpDelete);
  TF_RegisterKernelBuilder("SignalFramer", builder, status);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
