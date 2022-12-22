/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/kernels/lstm_eval_general.h"

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace lstm_internal {

// Increment the data offset so the sigle time step invocation call can access
// the corresponding input/output tensor data at the time step
void LstmStepManager::UpdateTime() {
  current_time_ += 1;
  TFLITE_DCHECK_LE(current_time_, size_info_.time_steps);
  // default as one batch per inference
  int input_step = size_info_.input_dimension;
  int output_step = size_info_.state_dimension;
  // time major: batch inference
  if (size_info_.time_major) {
    input_step = input_step * size_info_.batch_size;
    output_step = output_step * size_info_.batch_size;
  }

  input_offset_ += input_step;
  output_offset_ += output_step;
}

// Increment the data offset so the sigle time step invocation call can access
// the corresponding hidden/cell state tensor data at the time step (for single
// batch inference only)
void LstmStepManager::UpdateBatch() {
  current_batch_ += 1;
  TFLITE_DCHECK_LE(current_batch_, size_info_.batch_size);
  // batch inference for time major: no action needed
  if (size_info_.time_major) {
    return;
  }
  // otherwise: singe batch inference, go to the next batch
  hidden_state_offset_ += size_info_.state_dimension;
  cell_state_offset_ += size_info_.state_dimension;
}

// Input shape for each single time LSTM invocation.
// Multi-batch for time_major input
const RuntimeShape LstmStepManager::InputShape() const {
  int batch_size = 1;
  if (size_info_.time_major) {
    batch_size = size_info_.batch_size;
  }
  const int dims[2] = {batch_size, size_info_.input_dimension};
  const int32_t* dims_data = reinterpret_cast<const int32_t*>(dims);
  return RuntimeShape(2, dims_data);
}

// State shape (both hidden and cell) for each single time LSTM invocation.
// Multi-batch for time_major input
const RuntimeShape LstmStepManager::StateShape() const {
  int batch_size = 1;
  if (size_info_.time_major) {
    batch_size = size_info_.batch_size;
  }
  const int dims[2] = {batch_size, size_info_.state_dimension};
  const int32_t* dims_data = reinterpret_cast<const int32_t*>(dims);
  return RuntimeShape(2, dims_data);
}

}  // namespace lstm_internal
}  // namespace tflite
