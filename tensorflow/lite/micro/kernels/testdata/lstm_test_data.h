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
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
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
struct GateData {
  WeightType activation_weight[state_dimension * input_dimension];
  WeightType recurrent_weight[state_dimension * state_dimension];
  BiasType fused_bias[state_dimension];
  // Quantized model folded the zero point of activations into biases:
  // bias + zero_point * weight.
  // Note: folded bias is only required for the legacy 8x8->16 pass. Therefore
  // the data type is fixed here to avoid compilation errors (the computation of
  // folding does not support other types)
  int32_t activation_zp_folded_bias[state_dimension];
  int32_t recurrent_zp_folded_bias[state_dimension];
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

// A struct that holds the quantization settings for the LSTM node. Data
// members can be grouped into five parts.
// 1. Data types (activation,weight, cell, bias)
// 2. Non-linear activation (i.e., tanh and sigmoid) fixed point
// calculation settings
// 3. Input/output tensor quantization settings
// 4. Internal state (hidden and cell) quantization settings
// 5. Internal gate (forget, input, cell, output) settings
struct NodeQuantizationParameters {
  TfLiteType activation_type;
  TfLiteType weight_type;
  TfLiteType cell_type;
  TfLiteType bias_type;
  // Fixed point setting for integer nonlinear activation calculation
  double nonlinear_activation_input_scale;
  double nonlinear_activation_output_scale;
  // Quantization parameters for input/output
  TensorQuantizationParameters input;
  TensorQuantizationParameters output;
  // Quantization parameters for internal states
  TensorQuantizationParameters hidden_state;
  TensorQuantizationParameters cell_state;
  // Quantization parameters for gates
  GateQuantizationParameters forget_gate;
  GateQuantizationParameters input_gate;
  GateQuantizationParameters cell_gate;
  GateQuantizationParameters output_gate;
};

// Data structure that holds all the information to evaluate a LSTM kernel
// (mimic the LSTM node).
// Tensor Types:
// ActivationType defines the data type of input/output of the layer. The hidden
// state has the ActivationType as well since it is the layer output of the
// previous time.
// WeightType defines the weight data type inside the internal gates.
// BiasType defines the bias data type inside the internal gates. (normally the
// same type of MatMul accumulator).
// Tensor Shapes:
// The input to the layer has shape (batch_size,time_steps,input_dimension).
// Both the hidden state and cell state has shape (state_dimension, 1)
// The output of the layer has shape (batch_size,time_steps,state_dimension)
//  Note: state values can change through calls (stateful)
template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
class LstmNodeContent {
 public:
  LstmNodeContent(const LstmNodeContent& other) = default;
  LstmNodeContent& operator=(const LstmNodeContent& other) = default;
  // Use the general model setting (builtin data) and the four gates data to
  // construct the node content. Note the input, hidden state, and cell state
  // data is provided later for flexible testing (initialize as zero now)
  LstmNodeContent(
      const TfLiteUnidirectionalSequenceLSTMParams builtin_data,
      const GateData<WeightType, BiasType, input_dimension, state_dimension>
          forget_gate_params,
      const GateData<WeightType, BiasType, input_dimension, state_dimension>
          input_gate_params,
      const GateData<WeightType, BiasType, input_dimension, state_dimension>
          cell_gate_params,
      const GateData<WeightType, BiasType, input_dimension, state_dimension>
          output_gate_params)
      : builtin_data_(builtin_data),
        forget_gate_data_(forget_gate_params),
        input_gate_data_(input_gate_params),
        cell_gate_data_(cell_gate_params),
        output_gate_data_(output_gate_params) {
    InitializeTensors();
  }

  // Add quantization parameters (scale, zero point) to tensors
  // Only required for the integer kernel
  void AddQuantizationParameters(
      const NodeQuantizationParameters& quantization_params) {
    quantization_settings_ = quantization_params;
    // Input Tensor
    SetTensorQuantizationParam(kLstmInputTensor, quantization_params.input);
    // Forget Gate Tensors
    const auto& forget_gate_quant_param = quantization_params.forget_gate;
    SetTensorQuantizationParam(kLstmInputToForgetWeightsTensor,
                               forget_gate_quant_param.activation_weight);
    SetTensorQuantizationParam(kLstmRecurrentToForgetWeightsTensor,
                               forget_gate_quant_param.recurrent_weight);
    SetTensorQuantizationParam(kLstmForgetGateBiasTensor,
                               forget_gate_quant_param.bias);
    // Input Gate Tensors
    const auto& input_gate_quant_param = quantization_params.input_gate;
    SetTensorQuantizationParam(kLstmInputToInputWeightsTensor,
                               input_gate_quant_param.activation_weight);
    SetTensorQuantizationParam(kLstmRecurrentToInputWeightsTensor,
                               input_gate_quant_param.recurrent_weight);
    SetTensorQuantizationParam(kLstmInputGateBiasTensor,
                               input_gate_quant_param.bias);
    // Cell Gate Tensors
    const auto& cell_gate_quant_param = quantization_params.cell_gate;
    SetTensorQuantizationParam(kLstmInputToCellWeightsTensor,
                               cell_gate_quant_param.activation_weight);
    SetTensorQuantizationParam(kLstmRecurrentToCellWeightsTensor,
                               cell_gate_quant_param.recurrent_weight);
    SetTensorQuantizationParam(kLstmCellGateBiasTensor,
                               cell_gate_quant_param.bias);
    // Output Gate Tensors
    const auto& output_gate_quant_param = quantization_params.output_gate;
    SetTensorQuantizationParam(kLstmInputToOutputWeightsTensor,
                               output_gate_quant_param.activation_weight);
    SetTensorQuantizationParam(kLstmRecurrentToOutputWeightsTensor,
                               output_gate_quant_param.recurrent_weight);
    SetTensorQuantizationParam(kLstmOutputGateBiasTensor,
                               output_gate_quant_param.bias);
    // State Tensors
    SetTensorQuantizationParam(kLstmOutputStateTensor,
                               quantization_params.hidden_state);
    SetTensorQuantizationParam(kLstmCellStateTensor,
                               quantization_params.cell_state);
    // Output Tensor
    SetTensorQuantizationParam(24, quantization_params.output);
  }

  // Provide interface to set the input tensor values for flexible testing
  void SetInputData(const ActivationType* data) {
    std::memcpy(
        input_, data,
        batch_size * input_dimension * time_steps * sizeof(ActivationType));
    SetTensor(kLstmInputTensor, input_, input_size_);
  }
  const ActivationType* GetInputData() const { return input_; }

  // Provide interface to set the hidden state tensor values for flexible
  // testing
  void SetHiddenStateData(const ActivationType* data) {
    std::memcpy(hidden_state_, data,
                batch_size * state_dimension * sizeof(ActivationType));
  }
  ActivationType* GetHiddenStateData() { return hidden_state_; }

  // Provide interface to set the cell state tensor values for flexible
  // testing
  void SetCellStateData(const CellType* data) {
    std::memcpy(cell_state_, data,
                batch_size * state_dimension * sizeof(CellType));
  }
  CellType* GetCellStateData() { return cell_state_; }
  ActivationType* GetOutputData() { return output_; }

  // Internal tensors, see lstm_shared.h for tensor names
  TfLiteEvalTensor* GetEvalTensor(const int tensor_index) {
    auto valid_index = input_tensor_indices_[tensor_index + 1];
    if (valid_index < 0) {
      return nullptr;
    }
    return &eval_tensors_[tensor_index];
  }

  TfLiteTensor* GetTensors() { return tensors_; }

  // Required by the kernel runner
  TfLiteIntArray* KernelInputs() {
    return IntArrayFromInts(input_tensor_indices_);
  }
  // Required by the kernel runner
  TfLiteIntArray* KernelOutputs() {
    return IntArrayFromInts(output_tensor_indices_);
  }

  // Variable tensors (will be changed, can not be const)
  TfLiteEvalTensor* HiddenStateEvalTensor() {
    return &eval_tensors_[kLstmOutputStateTensor];
  }
  TfLiteEvalTensor* CellStateEvalTensor() {
    return &eval_tensors_[kLstmCellStateTensor];
  }
  TfLiteEvalTensor* OutputEvalTensor() { return &eval_tensors_[24]; }

  const GateData<WeightType, BiasType, input_dimension, state_dimension>&
  ForgetGateData() const {
    return forget_gate_data_;
  }
  const GateData<WeightType, BiasType, input_dimension, state_dimension>&
  InputGateData() const {
    return input_gate_data_;
  }
  const GateData<WeightType, BiasType, input_dimension, state_dimension>&
  CellGateData() const {
    return cell_gate_data_;
  }
  const GateData<WeightType, BiasType, input_dimension, state_dimension>&
  OutputGateData() const {
    return output_gate_data_;
  }

  const TfLiteUnidirectionalSequenceLSTMParams& BuiltinData() const {
    return builtin_data_;
  }

  const NodeQuantizationParameters& QuantizationSettings() const {
    return quantization_settings_;
  }

 private:
  void InitializeTensors() {
    // Invalid all the input tensors untill we set it
    input_tensor_indices_[0] = 24;  // tot elements
    for (size_t i = 1; i < 25; i++) {
      input_tensor_indices_[i] = kTfLiteOptionalTensor;
    }
    // Input Tensor
    SetTensor(kLstmInputTensor, input_, input_size_);
    // Forget Gate Tensors
    SetTensor(kLstmInputToForgetWeightsTensor,
              forget_gate_data_.activation_weight, activation_weight_size_);
    SetTensor(kLstmRecurrentToForgetWeightsTensor,
              forget_gate_data_.recurrent_weight, recurrent_weight_size_);
    SetTensor(kLstmForgetGateBiasTensor, forget_gate_data_.fused_bias,
              bias_size_);
    // Input Gate Tensors
    SetTensor(kLstmInputToInputWeightsTensor,
              input_gate_data_.activation_weight, activation_weight_size_);
    SetTensor(kLstmRecurrentToInputWeightsTensor,
              input_gate_data_.recurrent_weight, recurrent_weight_size_);
    SetTensor(kLstmInputGateBiasTensor, input_gate_data_.fused_bias,
              bias_size_);
    // Cell Gate Tensors
    SetTensor(kLstmInputToCellWeightsTensor, cell_gate_data_.activation_weight,
              activation_weight_size_);
    SetTensor(kLstmRecurrentToCellWeightsTensor,
              cell_gate_data_.recurrent_weight, recurrent_weight_size_);
    SetTensor(kLstmCellGateBiasTensor, cell_gate_data_.fused_bias, bias_size_);
    // Output Gate Tensors
    SetTensor(kLstmInputToOutputWeightsTensor,
              output_gate_data_.activation_weight, activation_weight_size_);
    SetTensor(kLstmRecurrentToOutputWeightsTensor,
              output_gate_data_.recurrent_weight, recurrent_weight_size_);
    SetTensor(kLstmOutputGateBiasTensor, output_gate_data_.fused_bias,
              bias_size_);
    // State Tensors
    SetTensor(kLstmOutputStateTensor, hidden_state_, state_size_,
              /*is_variable=*/true);
    SetTensor(kLstmCellStateTensor, cell_state_, state_size_,
              /*is_variable=*/true);
    // // Output Tensor
    SetTensor(24, output_, output_size_, /*is_variable=*/true);
  }

  template <typename T>
  void SetTensor(const int index, const T* data, int* dims,
                 const bool is_variable = false) {
    // Lite tensors for kernel level testing
    tensors_[index].data.data = const_cast<T*>(data);
    tensors_[index].dims = IntArrayFromInts(dims);
    tensors_[index].type = typeToTfLiteType<T>();
    tensors_[index].is_variable = is_variable;
    // Eval tensors for internal computation testing
    eval_tensors_[index].data.data = const_cast<T*>(data);
    eval_tensors_[index].dims = IntArrayFromInts(dims);
    eval_tensors_[index].type = typeToTfLiteType<T>();
    // update the index
    if (index < 24) {
      input_tensor_indices_[index + 1] = index;
    }
  }

  void SetTensorQuantizationParam(
      const int index, const TensorQuantizationParameters& quant_param) {
    tensors_[index].params.scale = quant_param.scale;
    tensors_[index].params.zero_point = quant_param.zero_point;
  }

  const TfLiteUnidirectionalSequenceLSTMParams builtin_data_;
  GateData<WeightType, BiasType, input_dimension, state_dimension>
      forget_gate_data_;
  GateData<WeightType, BiasType, input_dimension, state_dimension>
      input_gate_data_;
  GateData<WeightType, BiasType, input_dimension, state_dimension>
      cell_gate_data_;
  GateData<WeightType, BiasType, input_dimension, state_dimension>
      output_gate_data_;

  // Keep to ease the testing process (although all quantization information can
  // be obtained from individual tensors, they are well organized here and light
  // weighted)
  NodeQuantizationParameters quantization_settings_;

  // Not const since IntArrayFromInts takes int *; the first element of the
  // array must be the size of the array
  int input_size_[4] = {3, batch_size, time_steps, input_dimension};
  int output_size_[4] = {3, batch_size, time_steps, state_dimension};
  // weight tensor has C-style "row-major" memory ordering
  int activation_weight_size_[3] = {2, state_dimension, input_dimension};
  int recurrent_weight_size_[3] = {2, state_dimension, state_dimension};
  int bias_size_[2] = {1, state_dimension};
  int state_size_[3] = {2, batch_size, state_dimension};

  // see lstm_shared.h for tensor names, the last tensor is the output tensor
  TfLiteTensor tensors_[24 + 1];
  // Use for internel kernel testing
  TfLiteEvalTensor eval_tensors_[24 + 1];
  // indices for the tensors inside the node (required by kernel runner)
  int input_tensor_indices_[1 + 24] = {};
  // single output (last in the tensors array)
  int output_tensor_indices_[2] = {1, 24};

  // tennsor data
  // states are initialized to zero
  ActivationType hidden_state_[batch_size * state_dimension] = {};
  CellType cell_state_[batch_size * state_dimension] = {};
  // input is defined in the ModelContent (const across all derived models)
  ActivationType input_[batch_size * input_dimension * time_steps] = {};
  ActivationType output_[batch_size * state_dimension * time_steps] = {};
};

//  Converts floating point gate parameters to the corresponding quantized
//  version
template <typename WeightType, typename BiasType, int input_dimension,
          int state_dimension>
GateData<WeightType, BiasType, input_dimension, state_dimension>
CreateQuantizedGateData(
    const GateData<float, float, input_dimension, state_dimension>&
        gate_parameters,
    const TensorQuantizationParameters& input_quantization_params,
    const TensorQuantizationParameters& output_quantization_params,
    const GateQuantizationParameters& gate_quantization_params,
    const bool fold_zero_point) {
  GateData<WeightType, BiasType, input_dimension, state_dimension>
      quantized_gate_params;
  tflite::SymmetricQuantize(gate_parameters.activation_weight,
                            quantized_gate_params.activation_weight,
                            state_dimension * input_dimension,
                            gate_quantization_params.activation_weight.scale);
  tflite::SymmetricQuantize(gate_parameters.recurrent_weight,
                            quantized_gate_params.recurrent_weight,
                            state_dimension * state_dimension,
                            gate_quantization_params.recurrent_weight.scale);
  tflite::SymmetricQuantize(gate_parameters.fused_bias,
                            quantized_gate_params.fused_bias, state_dimension,
                            gate_quantization_params.bias.scale);
  // Note: steps below are not required for the generalized LSTM evaluation
  // (e.g., 16bits activation)
  if (fold_zero_point) {
    // Copy the bias values to prepare zero_point folded
    // bias precomputation. bias has same scale as
    // input_scale*input_weight_scale)
    std::memcpy(quantized_gate_params.activation_zp_folded_bias,
                quantized_gate_params.fused_bias, 2 * sizeof(int32_t));
    // Pre-calculate bias - zero_point * weight (a constant).
    tflite::tensor_utils::MatrixScalarMultiplyAccumulate(
        quantized_gate_params.activation_weight,
        -1 * input_quantization_params.zero_point, 2, 2,
        quantized_gate_params.activation_zp_folded_bias);

    // Initialize the folded bias to zeros for accumulation
    for (size_t i = 0; i < 2; i++) {
      quantized_gate_params.recurrent_zp_folded_bias[i] = 0;
    }
    // Calculate : -zero_point * weight since it is a constant
    tflite::tensor_utils::MatrixScalarMultiplyAccumulate(
        quantized_gate_params.recurrent_weight,
        -1 * output_quantization_params.zero_point, 2, 2,
        quantized_gate_params.recurrent_zp_folded_bias);
  }
  return quantized_gate_params;
}

// Create integer LSTM node content from the float node contents and
// quantization settings
// Note: fold_zero_point folds the zero point into the bias (precomputation),
// which is not required for the generalized integer inference (16 bits act
// LSTM).
template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                time_steps, input_dimension, state_dimension>
CreateIntegerNodeContents(
    const NodeQuantizationParameters& quantization_settings,
    const bool fold_zero_point,
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& float_node_contents) {
  const auto quantized_forget_gate_data =
      CreateQuantizedGateData<WeightType, BiasType, input_dimension,
                              state_dimension>(
          float_node_contents.ForgetGateData(), quantization_settings.input,
          quantization_settings.output, quantization_settings.forget_gate,
          fold_zero_point);
  const auto quantized_input_gate_data =
      CreateQuantizedGateData<WeightType, BiasType, input_dimension,
                              state_dimension>(
          float_node_contents.InputGateData(), quantization_settings.input,
          quantization_settings.output, quantization_settings.input_gate,
          fold_zero_point);
  const auto quantized_cell_gate_data =
      CreateQuantizedGateData<WeightType, BiasType, input_dimension,
                              state_dimension>(
          float_node_contents.CellGateData(), quantization_settings.input,
          quantization_settings.output, quantization_settings.cell_gate,
          fold_zero_point);
  const auto quantized_output_gate_params =
      CreateQuantizedGateData<WeightType, BiasType, input_dimension,
                              state_dimension>(
          float_node_contents.OutputGateData(), quantization_settings.input,
          quantization_settings.output, quantization_settings.output_gate,
          fold_zero_point);
  LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                  time_steps, input_dimension, state_dimension>
      quantized_node_content(
          float_node_contents.BuiltinData(), quantized_forget_gate_data,
          quantized_input_gate_data, quantized_cell_gate_data,
          quantized_output_gate_params);

  // Quantize the floating point input
  ActivationType quantized_input[batch_size * input_dimension * time_steps] =
      {};
  Quantize(float_node_contents.GetInputData(), quantized_input,
           batch_size * input_dimension * time_steps,
           quantization_settings.input.scale,
           quantization_settings.input.zero_point);
  quantized_node_content.SetInputData(quantized_input);
  // Quantize the  floating point hidden state
  ActivationType quantized_hidden_state[batch_size * state_dimension] = {};
  Quantize(float_node_contents.GetHiddenStateData(), quantized_hidden_state,
           batch_size * state_dimension,
           quantization_settings.hidden_state.scale,
           quantization_settings.hidden_state.zero_point);
  quantized_node_content.SetHiddenStateData(quantized_hidden_state);
  // Quantize the floating point cell state
  CellType quantized_cell_state[batch_size * state_dimension] = {};
  Quantize(float_node_contents.GetCellStateData(), quantized_cell_state,
           batch_size * state_dimension, quantization_settings.cell_state.scale,
           quantization_settings.cell_state.zero_point);
  quantized_node_content.SetCellStateData(quantized_cell_state);

  // Add scale and zero point to tensors
  quantized_node_content.AddQuantizationParameters(quantization_settings);
  return quantized_node_content;
}

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

// Create a 2x2 float node content
// batch_size = 2; time_steps = 3; input_dimension = 2; state_dimension = 2
LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
Create2x3x2X2FloatNodeContents(const float* input_data = nullptr,
                               const float* hidden_state = nullptr,
                               const float* cell_state = nullptr);

// Get the quantization settings for the 2X2 model
NodeQuantizationParameters Get2X2Int8LstmQuantizationSettings();

// Create int8 (activation) x int8 (weight) -> int16 (cell) node
// batch_size = 2; time_steps = 3; input_dimension = 2; state_dimension = 2
// input is in float format since the source of truth is always the float
// configuration
LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
Create2x3x2X2Int8NodeContents(const float* input_data = nullptr,
                              const float* hidden_state = nullptr,
                              const float* cell_state = nullptr);

// Create int16 (activation) x int8 (weight) -> int16 (cell) node
// batch_size = 2; time_steps = 3; input_dimension = 2; state_dimension = 2
// input is in float format since the source of truth is always the float
// configuration
LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2, 2>
Create2x3x2X2Int16NodeContents(const float* input_data = nullptr,
                               const float* hidden_state = nullptr,
                               const float* cell_state = nullptr);

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_TESTDATA_LSTM_TEST_DATA_H_
