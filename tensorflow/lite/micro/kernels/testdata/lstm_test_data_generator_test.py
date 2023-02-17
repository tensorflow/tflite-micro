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
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime
from tflite_micro.tensorflow.lite.micro.examples.mnist_lstm import evaluate

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


def create_keras_lstm(stateful=True):
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
  keras_lstm = create_keras_lstm()
