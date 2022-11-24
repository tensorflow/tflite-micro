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

#include <cstring>

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

// Struct that holds the weight/bias information for a standard gate (i.e. no
// modification such as layer normalization, peephole, etc.)
template <typename WeightType, typename BiasType, int input_dimension,
          int state_dimension>
struct GateParameters {
  WeightType activation_weight[state_dimension * input_dimension];
  WeightType recurrent_weight[state_dimension * state_dimension];
  BiasType fused_bias[state_dimension];
  // Quantized model folded the zero point of activations into biases:
  // bias + zero_point * weight.
  BiasType activation_zp_folded_bias[state_dimension];
  BiasType recurrent_zp_folded_bias[state_dimension];
};

// A base class that holds all the tensors for evaluation
template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
class TestModelContents {
 public:
  TestModelContents(const GateParameters<WeightType, BiasType, input_dimension,
                                         state_dimension>
                        forget_gate_params,
                    const GateParameters<WeightType, BiasType, input_dimension,
                                         state_dimension>
                        input_gate_params,
                    const GateParameters<WeightType, BiasType, input_dimension,
                                         state_dimension>
                        cell_gate_params,
                    const GateParameters<WeightType, BiasType, input_dimension,
                                         state_dimension>
                        output_gate_params);
  // Provide interface to set the input tensor values for flexible testing
  void SetInputTensorData(const ActivationType* data) {
    std::memcpy(
        input_, data,
        batch_size * input_dimension * time_steps * sizeof(ActivationType));
    SetTensor(0, input_, input_size_);
  }

  // Provide interface to set the hidden state tensor values for flexible
  // testing
  void SetHiddenStateTensorData(const ActivationType* data) {
    std::memcpy(hidden_state_, data,
                batch_size * state_dimension * sizeof(ActivationType));
    SetTensor(13, hidden_state_, state_size_);
  }

  TfLiteEvalTensor* GetTensor(int tensor_index) {
    return &tensors_[tensor_index];
  }
  const ActivationType* GetHiddenState() const { return hidden_state_; }
  const CellType* GetCellState() const { return cell_state_; }
  const ActivationType* GetOutput() const { return output_; }

  CellType* ScratchBuffers() { return scratch_buffers_; }

  // TODO(b/253466487): make all getters constant after refactor the
  // IntegerLstmParameter
  GateParameters<WeightType, BiasType, input_dimension, state_dimension>&
  ForgetGateParams() {
    return forget_gate_params_;
  }
  GateParameters<WeightType, BiasType, input_dimension, state_dimension>&
  InputGateParams() {
    return input_gate_params_;
  }
  GateParameters<WeightType, BiasType, input_dimension, state_dimension>&
  CellGateParams() {
    return cell_gate_params_;
  }
  GateParameters<WeightType, BiasType, input_dimension, state_dimension>&
  OutputGateParams() {
    return output_gate_params_;
  }

 private:
  template <typename T>
  void SetTensor(const int index, const T* data, int* dims);

  GateParameters<WeightType, BiasType, input_dimension, state_dimension>
      forget_gate_params_;
  GateParameters<WeightType, BiasType, input_dimension, state_dimension>
      input_gate_params_;
  GateParameters<WeightType, BiasType, input_dimension, state_dimension>
      cell_gate_params_;
  GateParameters<WeightType, BiasType, input_dimension, state_dimension>
      output_gate_params_;

  // Not const since IntArrayFromInts takes int *; the first element of the
  // array must be the size of the array
  int input_size_[4] = {3, batch_size, time_steps, input_dimension};
  int output_size_[4] = {3, batch_size, time_steps, state_dimension};
  int activation_weight_size_[3] = {2, state_dimension, input_dimension};
  int recurrent_weight_size_[3] = {2, state_dimension, state_dimension};
  int bias_size_[3] = {2, batch_size, state_dimension};
  int state_size_[3] = {2, batch_size, state_dimension};

  // 0 input; 1-12 gate parameters; 13-14 states; 15 output
  TfLiteEvalTensor tensors_[16];

  // states are initialized to zero
  ActivationType hidden_state_[batch_size * state_dimension] = {};
  CellType cell_state_[batch_size * state_dimension] = {};
  // input is defined in the ModelContent (const across all derived models)
  ActivationType input_[batch_size * input_dimension * time_steps] = {};
  ActivationType output_[batch_size * state_dimension * time_steps] = {};
  // scratch buffers (4; used for floating point model only)
  CellType scratch_buffers_[4 * batch_size * state_dimension] = {};
};

// A struct that holds quantization parameters for a LSTM Tensor
struct TensorQuantizationParameters {
  TensorQuantizationParameters() = default;
  TensorQuantizationParameters(const double arg_scale, const int arg_zero_point,
                               const bool arg_symmetry)
      : scale(arg_scale), zero_point(arg_zero_point), symmetry(arg_symmetry) {}
  // all the effective
  double scale = 0;
  int zero_point = 0;
  bool symmetry = false;
};

struct GateQuantizationParameters {
  GateQuantizationParameters() = default;
  GateQuantizationParameters(
      const TensorQuantizationParameters arg_activation_weight,
      const TensorQuantizationParameters arg_recurrent_weight,
      const TensorQuantizationParameters arg_bias)
      : activation_weight(arg_activation_weight),
        recurrent_weight(arg_recurrent_weight),
        bias(arg_bias) {}
  TensorQuantizationParameters activation_weight;
  TensorQuantizationParameters recurrent_weight;
  TensorQuantizationParameters bias;
};

// A struct that holds the quantization settings for the model
struct ModelQuantizationParameters {
  TfLiteType activation_type;
  TfLiteType cell_type;
  TfLiteType bias_type;
  double nonlinear_activation_input_scale;
  double nonlinear_activation_output_scale;
  // Quantization parameters for input/output
  TensorQuantizationParameters input_quantization_parameters;
  TensorQuantizationParameters output_quantization_parameters;
  // Quantization parameters for internal states
  TensorQuantizationParameters hidden_quantization_parameters;
  TensorQuantizationParameters cell_quantization_parameters;
  // Quantization parameters for gates
  GateQuantizationParameters forget_gate_quantization_parameters;
  GateQuantizationParameters input_gate_quantization_parameters;
  GateQuantizationParameters cell_gate_quantization_parameters;
  GateQuantizationParameters output_gate_quantization_parameters;
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

// Create a 2x2 float model content
// batch_size = 2; time_steps = 3; input_dimension = 2; state_dimension = 2
TestModelContents<float, float, float, float, 2, 3, 2, 2>
Create2x3x2X2FloatModelContents();

// Get the quantization settings for the 2X2 model
ModelQuantizationParameters Get2X2Int8LstmQuantizationSettings();

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_TESTDATA_LSTM_TEST_DATA_H_
