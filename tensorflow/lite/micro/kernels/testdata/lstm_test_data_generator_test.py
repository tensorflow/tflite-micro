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
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.kernels.testdata import lstm_test_data_utils

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
        'activation_weight_data': [1, 1, 1, 1],
        'recurrent_weight_data': [1, 1, 1, 1],
        'bias_data': [0, 0],
    },
    'input_gate_data': {
        'activation_weight_data': [1, 1, 1, 1],
        'recurrent_weight_data': [1, 1, 1, 1],
        'bias_data': [0, 0],
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

_KERNEL_INITIALIZATION_SETTINGS = {
    'init_hidden_state_vals': [0, 0],
    'init_cell_state_vals': [0, 0],
    'hidden_state_range': (-1, 1),
    'cell_state_range': [-8, 8],
}


def create_keras_lstm(stateful=True):
  """Create a keras model with LSTM layer only for testing"""
  input_layer = tf.keras.layers.Input(shape=(1, 2), batch_size=1, name="input")
  lstm_output = tf.keras.layers.LSTM(units=2,
                                     return_sequences=True,
                                     stateful=stateful,
                                     unit_forget_bias=False,
                                     return_state=True,
                                     kernel_initializer="ones",
                                     recurrent_initializer="ones",
                                     bias_initializer="zeros")(input_layer)
  return tf.keras.Model(input_layer, lstm_output, name="LSTM")


class QuantizedLSTMDebuggerTest(test_util.TensorFlowTestCase):

  # only the float output from the debugger is used to setup the test data in .cc
  def testFloatCompareWithKeras(self):
    keras_lstm = create_keras_lstm()
    lstm_debugger = lstm_test_data_utils.QuantizedLSTMDebugger(
        _KERNEL_CONFIG,
        _KERNEL_PARAMETERS,
        _KERNEL_INITIALIZATION_SETTINGS['init_hidden_state_vals'],
        _KERNEL_INITIALIZATION_SETTINGS['hidden_state_range'],
        _KERNEL_INITIALIZATION_SETTINGS['init_cell_state_vals'],
        _KERNEL_INITIALIZATION_SETTINGS['cell_state_range'],
    )

    num_steps = 20
    for _ in range(num_steps):
      # debugger has input shape (input_dim, 1)
      test_data = np.random.rand(2, 1)
      input_tensor = lstm_test_data_utils.assemble_quantized_tensor(
          test_data, -1, 1, False)
      _, output_float = lstm_debugger.invoke(input_tensor)
      output_keras, _, _ = keras_lstm.predict(test_data.reshape(1, 1, 2))

      diff = abs(output_float.flatten() - output_keras.flatten())
      self.assertAllLess(diff, 1e-6)


if __name__ == "__main__":
  test.main()