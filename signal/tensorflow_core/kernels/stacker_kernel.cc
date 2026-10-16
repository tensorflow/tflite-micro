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

struct StackerOp {
  int32_t num_channels = 0;
  int32_t stacker_left_context = 0;
  int32_t stacker_right_context = 0;
  int32_t stacker_step = 0;
  size_t buffer_size = 0;
  size_t step_size = 0;
  bool stacker_has_first_frame = false;
  std::vector<int8_t> state;
  tflite::tflm_signal::CircularBuffer* circular_buffer = nullptr;
};

void* StackerOpCreate(TF_OpKernelConstruction* ctx) {
  auto* op = new StackerOp{};
  TF_Status* status = TF_NewStatus();

  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_channels", &op->num_channels,
                                       status);
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrInt32(ctx, "stacker_left_context",
                                         &op->stacker_left_context, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrInt32(ctx, "stacker_right_context",
                                         &op->stacker_right_context, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrInt32(ctx, "stacker_step", &op->stacker_step,
                                         status);
  }

  if (TF_GetCode(status) == TF_OK) {
    op->buffer_size =
        static_cast<size_t>(op->num_channels * (op->stacker_left_context +
                                                op->stacker_right_context + 1));
    op->step_size = static_cast<size_t>(op->num_channels * op->stacker_step);
    op->stacker_has_first_frame = false;

    size_t state_size =
        tflite::tflm_signal::CircularBufferGetNeededMemory(op->buffer_size);
    op->state.resize(state_size);
    op->circular_buffer = tflite::tflm_signal::CircularBufferInit(
        op->buffer_size, op->state.data(), state_size);
  } else {
    TF_OpKernelConstruction_Failure(ctx, status);
    delete op;
    op = nullptr;
  }

  TF_DeleteStatus(status);
  return op;
}

void StackerOpDelete(void* kernel) { delete static_cast<StackerOp*>(kernel); }

void StackerOpCompute(void* kernel, TF_OpKernelContext* ctx) {
  auto* op = static_cast<StackerOp*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;

  TF_GetInput(ctx, 0, &input_tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    const auto* input =
        static_cast<const int16_t*>(TF_TensorData(input_tensor));

    tflite::tflm_signal::CircularBufferWrite(
        op->circular_buffer, input, static_cast<size_t>(op->num_channels));

    // The first frame is replicated an extra left_context times to pad.
    if (!op->stacker_has_first_frame) {
      tflite::tflm_signal::CircularBufferExtend(
          op->circular_buffer, static_cast<size_t>(op->num_channels),
          op->stacker_left_context);
      op->stacker_has_first_frame = true;
    }

    int64_t out_dims[1] = {static_cast<int64_t>(op->buffer_size)};
    TF_Tensor* output_tensor =
        TF_AllocateOutput(ctx, 0, TF_INT16, out_dims, 1,
                          op->buffer_size * sizeof(int16_t), status);
    TF_Tensor* output_valid_tensor = nullptr;
    if (TF_GetCode(status) == TF_OK) {
      output_valid_tensor =
          TF_AllocateOutput(ctx, 1, TF_BOOL, nullptr, 0, sizeof(bool), status);
    }

    if (TF_GetCode(status) == TF_OK) {
      auto* output = static_cast<int16_t*>(TF_TensorData(output_tensor));
      auto* output_valid =
          static_cast<bool*>(TF_TensorData(output_valid_tensor));

      if (tflite::tflm_signal::CircularBufferAvailable(op->circular_buffer) >=
          op->buffer_size) {
        tflite::tflm_signal::CircularBufferGet(op->circular_buffer,
                                               op->buffer_size, output);
        tflite::tflm_signal::CircularBufferDiscard(op->circular_buffer,
                                                   op->step_size);
        *output_valid = true;
      } else {
        *output_valid = false;
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
void RegisterSignalStackerKernel() {
  TF_Status* status = TF_NewStatus();
  auto* builder = TF_NewKernelBuilder("SignalStacker", "CPU", &StackerOpCreate,
                                      &StackerOpCompute, &StackerOpDelete);
  TF_RegisterKernelBuilder("SignalStacker", builder, status);
  TF_DeleteStatus(status);
}

const bool kSignalStackerKernelRegistered = []() {
  RegisterSignalStackerKernel();
  return true;
}();

}  // namespace
}  // namespace signal
}  // namespace tensorflow
