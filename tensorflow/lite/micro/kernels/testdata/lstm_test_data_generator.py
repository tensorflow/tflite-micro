import tensorflow as tf
from absl import app
import numpy as np

from tflite_micro.tensorflow.lite.micro.kernels.testdata.lstm_test_data_utils import *

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

_GATE_TEST_DATA = {
    'init_hidden_state_vals': [-0.1, 0.2],
    'init_cell_state_vals': [-1.3, 6.2],
    'input_data': [0.2, 0.3],
    'hidden_state_range': (-0.5, 0.7),
    'cell_state_range': [-8, 8],
    'input_data_range': [-1, 1]
}

_MULTISTEP_TEST_DATA = {
    'init_hidden_state_vals': [0, 0],
    'init_cell_state_vals': [0, 0],
    'input_data': [0.2, 0.3, 0.2, 0.3, 0.2, 0.3],  # three time steps 
    'hidden_state_range': (-0.5, 0.7),
    'cell_state_range': [-8, 8],
    'input_data_range': [-1, 1]
}


def one_step():
  lstm_debugger = QuantizedLSTMDebugger(
      _KERNEL_CONFIG,
      _KERNEL_PARAMETERS,
      _GATE_TEST_DATA['init_hidden_state_vals'],
      _GATE_TEST_DATA['hidden_state_range'],
      _GATE_TEST_DATA['init_cell_state_vals'],
      _GATE_TEST_DATA['cell_state_range'],
  )

  test_data = np.array(_GATE_TEST_DATA['input_data']).reshape((-1, 1))
  input_data_range = _GATE_TEST_DATA['input_data_range']
  input_tensor = assemble_quantized_tensor(test_data, input_data_range[0],
                                           input_data_range[1], False)
  lstm_debugger.invoke(input_tensor, debug=True)


def multi_step():
  lstm_debugger = QuantizedLSTMDebugger(
      _KERNEL_CONFIG,
      _KERNEL_PARAMETERS,
      _MULTISTEP_TEST_DATA['init_hidden_state_vals'],
      _MULTISTEP_TEST_DATA['hidden_state_range'],
      _MULTISTEP_TEST_DATA['init_cell_state_vals'],
      _MULTISTEP_TEST_DATA['cell_state_range'],
  )
  input_data = _MULTISTEP_TEST_DATA['input_data']
  input_data_range = _MULTISTEP_TEST_DATA['input_data_range']
  input_data_size = _KERNEL_CONFIG['shape_info']['input_dim']
  input_start_pos = 0
  while input_start_pos < len(input_data):
    one_step_data = np.array(input_data[input_start_pos:input_start_pos +
                                        input_data_size]).reshape((-1, 1))
    input_tensor = assemble_quantized_tensor(one_step_data,
                                             input_data_range[0],
                                             input_data_range[1], False)
    lstm_debugger.invoke(input_tensor, debug=True)
    input_start_pos += input_data_size


def main(_):
  one_step()
  multi_step()


if __name__ == "__main__":
  app.run(main)
