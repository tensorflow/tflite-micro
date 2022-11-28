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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_TESTDATA_LSTM_TEST_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_TESTDATA_LSTM_TEST_DATA_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace testing {
// Data structure to store all the data used to check output of internal gates
// of one time step
// input_size = batch_size*input_dimension gate_output_size =
// batch_size*state_dimension
template <int input_size, int gate_output_size>
struct GateOutputCheckData {
  float input_data[input_size];
  float hidden_state[gate_output_size];
  float cell_state[gate_output_size];
  float expected_forget_gate_output[gate_output_size];
  float expected_input_gate_output[gate_output_size];
  float expected_output_gate_output[gate_output_size];
  float expected_cell_gate_output[gate_output_size];
  float expected_updated_cell[gate_output_size];
  float expected_updated_hidden[gate_output_size];
};

// Data structure to store all the data used to check the output of the kernel
// of multiple batch, multiple timesteps
// input_size = batch_size*time_steps*input_dimension
// gate_output_size = batch_size*state_dimension
// output_size = time_steps*gate_output_size
template <int input_size, int gate_output_size, int output_size>
struct LstmEvalCheckData {
  float input_data[input_size];
  float hidden_state[gate_output_size];
  float expected_output[output_size];
  float expected_hidden_state[gate_output_size];
  float expected_cell_state[gate_output_size];
};

// Get the gate output data (one time step) for a simple 2X2 model
// batch_size = 2; time_steps = 1; input_dimension = 2; state_dimension = 2
// input_size = batch_size*time_steps*input_dimension = 4
// gate_output_size = batch_size*state_dimension = 4
GateOutputCheckData<4, 4> Get2X2GateOutputCheckData();

// Get the kernel output data for a simple 2X2 model
// batch_size = 2; time_steps = 3; input_dimension = 2; state_dimension = 2
// input_size = batch_size*time_steps*input_dimension = 12
// gate_output_size = batch_size*state_dimension = 4
// output_size = time_steps*gate_output_size = 12
LstmEvalCheckData<12, 4, 12> Get2X2LstmEvalCheckData();

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_TESTDATA_LSTM_TEST_DATA_H_
