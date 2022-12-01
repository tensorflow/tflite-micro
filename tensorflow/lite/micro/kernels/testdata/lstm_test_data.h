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
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/test_helpers.h"

namespace tflite {
namespace testing {
// Data structure to store all the data used to check output of internal gates
// of one time step
// input_size = batch_size*input_dimension (size of the input array)
// gate_output_size = batch_size*state_dimension (size of the gate output)
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
// input_size = batch_size*time_steps*input_dimension (size of the input array)
// gate_output_size = batch_size*state_dimension (size of the gate output)
// output_size = time_steps*gate_output_size (size of the output from the
// kernel)
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
// Every gate is defined by the type and size of the weights (bias included)
// inside.
// Specifically, types are weight type and bias type (normally the same
// type of MatMul accumulator).
// activation_weight has shape (hidden state dimension * input tensor dimension)
// recurrent_weight has shape (hidden state dimension * hidden state dimension)
// bias has shape (hidden state dimension, 1)
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

// Data structure that holds all the information to evaluate a LSTM kernel.
// ActivationType defines the data type of input/output of the layer. The hidden
// state has the ActivationType as well since it is the layer output of the
// previous time
// WeightType defines the weight data type inside the internal gates.
// BiasType defines the bias data type inside the internal gates. (normally the
// same type of MatMul accumulator).
// The input to the layer has shape (batch_size,time_steps,input_dimension)
// Both the hidden state and cell state has shape (state_dimension, 1)
// The output of the layer has shape (batch_size,time_steps,state_dimension)
// Note it is non-const since LSTM is stateful
// TODO: Seprate tensors into weights/states/io and use the name from
// lstm_common.h
template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
class ModelContents {
 public:
  ModelContents(const ModelContents& other) = default;
  ModelContents& operator=(const ModelContents& other) = default;

  ModelContents(const GateParameters<WeightType, BiasType, input_dimension,
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
                    output_gate_params)
      : forget_gate_params_(forget_gate_params),
        input_gate_params_(input_gate_params),
        cell_gate_params_(cell_gate_params),
        output_gate_params_(output_gate_params) {
    InitializeTensors();
  }

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

  // Provide interface to set the hidden state tensor values for flexible
  // testing
  void SetCellStateTensorData(const CellType* data) {
    std::memcpy(cell_state_, data,
                batch_size * state_dimension * sizeof(CellType));
    SetTensor(14, cell_state_, state_size_);
  }

  // I/O and states are changeable (not const)
  TfLiteEvalTensor* GetTensor(const int tensor_index) {
    return &tensors_[tensor_index];
  }

  // I/O and states are changeable (not const)
  ActivationType* GetHiddenState() { return hidden_state_; }
  CellType* GetCellState() { return cell_state_; }
  ActivationType* GetInput() { return input_; }
  ActivationType* GetOutput() { return output_; }

  // IntegerLstmParameter
  const GateParameters<WeightType, BiasType, input_dimension, state_dimension>&
  ForgetGateParams() const {
    return forget_gate_params_;
  }
  const GateParameters<WeightType, BiasType, input_dimension, state_dimension>&
  InputGateParams() const {
    return input_gate_params_;
  }
  const GateParameters<WeightType, BiasType, input_dimension, state_dimension>&
  CellGateParams() const {
    return cell_gate_params_;
  }
  const GateParameters<WeightType, BiasType, input_dimension, state_dimension>&
  OutputGateParams() const {
    return output_gate_params_;
  }

 private:
  void InitializeTensors() {
    // Input Tensor
    SetTensor(0, input_, input_size_);
    // Forget Gate Tensors
    SetTensor(1, forget_gate_params_.activation_weight,
              activation_weight_size_);
    SetTensor(2, forget_gate_params_.recurrent_weight, recurrent_weight_size_);
    SetTensor(3, forget_gate_params_.fused_bias, bias_size_);
    // Input Gate Tensors
    SetTensor(4, input_gate_params_.activation_weight, activation_weight_size_);
    SetTensor(5, input_gate_params_.recurrent_weight, recurrent_weight_size_);
    SetTensor(6, input_gate_params_.fused_bias, bias_size_);
    // Cell Gate Tensors
    SetTensor(7, cell_gate_params_.activation_weight, activation_weight_size_);
    SetTensor(8, cell_gate_params_.recurrent_weight, recurrent_weight_size_);
    SetTensor(9, cell_gate_params_.fused_bias, bias_size_);
    // Output Gate Tensors
    SetTensor(10, output_gate_params_.activation_weight,
              activation_weight_size_);
    SetTensor(11, output_gate_params_.recurrent_weight, recurrent_weight_size_);
    SetTensor(12, output_gate_params_.fused_bias, bias_size_);
    // State Tensors
    SetTensor(13, hidden_state_, state_size_);
    SetTensor(14, cell_state_, state_size_);
    // Output Tensor
    SetTensor(15, output_, output_size_);
  }

  template <typename T>
  void SetTensor(const int index, const T* data, int* dims) {
    tensors_[index].data.data = const_cast<T*>(data);
    tensors_[index].dims = IntArrayFromInts(dims);
    tensors_[index].type = typeToTfLiteType<T>();
  }

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
};

// A struct that holds quantization parameters for a LSTM Tensor
struct TensorQuantizationParameters {
  double scale;
  int zero_point;
  bool symmetry;
};

// A struct that holds quantization parameters for an internal gate, which is
// defined by activation/recurrent weight and bias (assuming no internal layer
// normalization)
struct GateQuantizationParameters {
  TensorQuantizationParameters activation_weight;
  TensorQuantizationParameters recurrent_weight;
  TensorQuantizationParameters bias;
};

// A struct that holds the quantization settings for the LSTM kernel. Data
// members can be grouped into five parts.
// 1. Data types (activation,weight, cell, bias)
// 2. Non-linear activation (i.e., tanh and sigmoid) fixed point
// calculation settings
// 3. Input/output tensor quantization settings
// 4. Internal state (hidden and cell) quantization settings
// 5. Internal gate (forget, input, cell, output) settings
struct ModelQuantizationParameters {
  TfLiteType activation_type;
  TfLiteType weight_type;
  TfLiteType cell_type;
  TfLiteType bias_type;
  // Fixed point setting for integer nonlinear activation calculation
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
ModelContents<float, float, float, float, 2, 3, 2, 2>
Create2x3x2X2FloatModelContents(const float* input_data = nullptr,
                                const float* hidden_state = nullptr,
                                const float* cell_state = nullptr);

// Get the quantization settings for the 2X2 model
ModelQuantizationParameters Get2X2Int8LstmQuantizationSettings();

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_TESTDATA_LSTM_TEST_DATA_H_
