# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
""" Generate the LSTM kernel test data settings in lstm_test_data.cc
1. Print the quantization settings for the test model (Get2X2Int8LstmQuantizationSettings in .cc)
2. Print the intermediate step outputs inside the LSTM for a single step LSTM invocation (Get2X2GateOutputCheckData in .cc)
3. Print the outputs for multi-step LSTM invocation (Get2X2LstmEvalCheckData in .cc)

Every invocation gives three types information:
1. Quantized output: kernel output in integer
2. Dequantized output: Quantized output in floating point representation
3. Float output: output from the floating point computation (i.e., float kernel)

Note:
1. Change quantization settings in _KERNEL_CONFIG to see the outcomes from various quantization schema (e.g., 8x8 Vs. 16x8)
2. Only single batch inference is supporte here. Change _GATE_TEST_DATA or _MULTISTEP_TEST_DATA to see kernel outputs on different input data
3. The quantization computation here is not the exact as the c++ implementation. The integer calculation is emulated here using floating point.
No fixed point math is implemented here. The purpose is to illustrate the computation procedure and possible quantization error accumulation, not for bit exactness.
"""
from absl import app
import numpy as np

from tflite_micro.tensorflow.lite.micro.kernels.testdata import lstm_test_data_utils

# Basic kernel information (defaul a 2x2 model with int8 quantization)
# change activation_bits to 16 for 16x8 case
_KERNEL_CONFIG = {
    'quantization_settings': {
        'weight_bits': 8,
        'activation_bits': 8,
        'bias_bits': 32,
        'cell_bits': 16,
    },
    'shape_info': {
        'input_dim': 2,
        'state_dim': 2
    }
}

# Kernel data setting (weight data for every gate). Corresponds to Create2x3x2X2FloatNodeContents in .cc
_KERNEL_PARAMETERS = {
    'forget_gate_data': {
        'activation_weight_data': [-10, -10, -20, -20],
        'recurrent_weight_data': [-10, -10, -20, -20],
        'bias_data': [1, 2],
    },
    'input_gate_data': {
        'activation_weight_data': [10, 10, 20, 20],
        'recurrent_weight_data': [10, 10, 20, 20],
        'bias_data': [-1, -2],
    },
    'cell_gate_data': {
        'activation_weight_data': [1, 1, 1, 1],
        'recurrent_weight_data': [1, 1, 1, 1],
        'bias_data': [0, 0],
    },
    'output_gate_data': {
        'activation_weight_data': [1, 1, 1, 1],
        'recurrent_weight_data': [1, 1, 1, 1],
        'bias_data': [0, 0],
    },
}

# Input and states setting for gate level testing (Get2X2GateOutputCheckData in .cc)
# Only single batch inference is supported (default as batch1 in .cc)
_GATE_TEST_DATA = {
    'init_hidden_state_vals': [-0.1, 0.2],
    'init_cell_state_vals': [-1.3, 6.2],
    'input_data': [0.2, 0.3],
    'hidden_state_range': (-0.5, 0.7),
    'cell_state_range': [-8, 8],
    'input_data_range': [-1, 1]
}

# Input and states setting for multi-step kernel testing (Get2X2LstmEvalCheckData in .cc)
# Only single batch inference is supported (default as batch1 in .cc)
_MULTISTEP_TEST_DATA = {
    'init_hidden_state_vals': [0, 0],
    'init_cell_state_vals': [0, 0],
    'input_data': [0.2, 0.3, 0.2, 0.3, 0.2, 0.3],  # three time steps
    'hidden_state_range': (-0.5, 0.7),
    'cell_state_range': [-8, 8],
    'input_data_range': [-1, 1]
}


def print_tensor_quantization_params(tensor_name, tensor):
  """Print the tensor quantization information (scale and zero point)"""
  print(f"{tensor_name}, scale: {tensor.scale}, zero_point:"
        f" {tensor.zero_point}")


def print_gate_tensor_params(gate_name, gate):
  """Print the quantization information for a gate (input/forget/cell/output gate)"""
  print(f"###### Quantization settings for {gate_name} ######")
  print_tensor_quantization_params("activation weight", gate.activation_weight)
  print_tensor_quantization_params("recurrent weight", gate.activation_weight)


def print_quantization_settings(lstm_debugger):
  """Print the quantization information for a LSTM kernel"""
  print_gate_tensor_params("forget gate", lstm_debugger.forget_gate_params)
  print_gate_tensor_params("input gate", lstm_debugger.input_gate_params)
  print_gate_tensor_params("cell gate", lstm_debugger.modulation_gate_params)
  print_gate_tensor_params("output gate", lstm_debugger.output_gate_params)
  print("###### State Tensors ######")
  print_tensor_quantization_params("Hidden State Tensor",
                                   lstm_debugger.hidden_state_tensor)
  print_tensor_quantization_params("Cell State Tensor",
                                   lstm_debugger.cell_state_tensor)


def print_one_step(lstm_debugger):
  """Print the intermediate calculation results for one step LSTM invocation (Get2X2GateOutputCheckData in .cc)"""
  test_data = np.array(_GATE_TEST_DATA['input_data']).reshape((-1, 1))
  input_data_range = _GATE_TEST_DATA['input_data_range']
  input_tensor = lstm_test_data_utils.assemble_quantized_tensor(
      test_data,
      input_data_range[0],
      input_data_range[1],
      symmetry=False,
      num_bits=_KERNEL_CONFIG['quantization_settings']['activation_bits'])
  lstm_debugger.invoke(input_tensor, debug=True)


def print_multi_step(lstm_debugger, debug=False):
  """Print the output of every step for multi step LSTM invocation (Get2X2LstmEvalCheckData in .cc)"""
  input_data = _MULTISTEP_TEST_DATA['input_data']
  input_data_range = _MULTISTEP_TEST_DATA['input_data_range']
  input_data_size = _KERNEL_CONFIG['shape_info']['input_dim']
  input_start_pos = 0
  steps = 0
  while input_start_pos < len(input_data):
    one_step_data = np.array(input_data[input_start_pos:input_start_pos +
                                        input_data_size]).reshape((-1, 1))
    input_tensor = lstm_test_data_utils.assemble_quantized_tensor(
        one_step_data,
        input_data_range[0],
        input_data_range[1],
        symmetry=False,
        num_bits=_KERNEL_CONFIG['quantization_settings']['activation_bits'])
    output_quant, output_float = lstm_debugger.invoke(input_tensor,
                                                      debug=debug)
    print(f"##### Step: {steps} #####")
    print(f"Quantized Output: {output_quant.flatten()}")
    print(
        f"Dequantized Output: {lstm_debugger.hidden_state_tensor.dequantized_data.flatten().flatten()}"
    )
    print(f"Float Output: {output_float.flatten()}")
    input_start_pos += input_data_size
    steps += 1


def main(_):
  one_step_lstm_debugger = lstm_test_data_utils.QuantizedLSTMDebugger(
      _KERNEL_CONFIG,
      _KERNEL_PARAMETERS,
      _GATE_TEST_DATA['init_hidden_state_vals'],
      _GATE_TEST_DATA['hidden_state_range'],
      _GATE_TEST_DATA['init_cell_state_vals'],
      _GATE_TEST_DATA['cell_state_range'],
  )
  print("========== Quantization Settings for the Test Kernal ========== ")
  print_quantization_settings(one_step_lstm_debugger)
  print("========== Single Step Invocation Intermediates  ========== ")
  print_one_step(one_step_lstm_debugger)

  multi_step_lstm_debugger = lstm_test_data_utils.QuantizedLSTMDebugger(
      _KERNEL_CONFIG,
      _KERNEL_PARAMETERS,
      _MULTISTEP_TEST_DATA['init_hidden_state_vals'],
      _MULTISTEP_TEST_DATA['hidden_state_range'],
      _MULTISTEP_TEST_DATA['init_cell_state_vals'],
      _MULTISTEP_TEST_DATA['cell_state_range'],
  )
  print("========== Multi Step Invocation Intermediates  ========== ")
  print_multi_step(multi_step_lstm_debugger)


if __name__ == "__main__":
  app.run(main)
