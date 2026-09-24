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
import unittest
import numpy as np

from tflite_micro.tensorflow.lite.micro.tools import requantize_flatbuffer
from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.tools import flatbuffer_utils


def create_simple_fc_model():
  """Create a simple model with two fully connected(fc) layers."""
  import tensorflow as tf

  model = tf.keras.models.Sequential(
    [
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="output"),
    ]
  )
  fixed_input = tf.keras.layers.Input(
    shape=[28, 28],
    batch_size=1,
    dtype=model.inputs[0].dtype,
    name="fixed_input",
  )
  fixed_output = model(fixed_input)
  return tf.keras.models.Model(fixed_input, fixed_output)


def representative_dataset_gen(num_samples=100):
  np.random.seed(42)
  for _ in range(num_samples):
    yield [np.random.random((1, 28, 28)).astype(np.float32)]


def convert_tfl_converter(keras_model, representative_dataset_gen, int16=False):
  """Convert and quantize the keras model using the standard tflite converter."""
  import tensorflow as tf

  converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  if int16:
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
  converter.representative_dataset = representative_dataset_gen
  converter._experimental_disable_per_channel_quantization_for_dense_layers = (
    True
  )
  converter._experimental_disable_per_channel = True
  return converter.convert()


def convert_8to16_requantizer(int8_model_bytes):
  '''Convert and quantize the int8 model using the int8 to int16 conversion tool'''
  int8_model = flatbuffer_utils.convert_bytearray_to_object(int8_model_bytes)
  # Use the tool to convert to int16
  requantizer = requantize_flatbuffer.Requantizer(int8_model)
  requantizer.requantize_8to16()
  return flatbuffer_utils.convert_object_to_bytearray(requantizer.model)


class SimpleFCModelTest(unittest.TestCase):
  def testCompareWithStandardConversion(self):

    def inference(tflm_interpreter, data_x):
      tflm_interpreter.set_input(data_x, 0)
      tflm_interpreter.invoke()
      return tflm_interpreter.get_output(0)

    dir_path = os.path.dirname(__file__)
    int16_path = os.path.join(dir_path, "simple_fc_int16.tflite")
    int8_path = os.path.join(dir_path, "simple_fc_int8.tflite")

    with open(int16_path, "rb") as f:
      tfl_converted_int16_model = f.read()

    with open(int8_path, "rb") as f:
      int8_model = f.read()

    int8_converted_int16_model = convert_8to16_requantizer(int8_model)

    interpreter_tfl_converted = runtime.Interpreter.from_bytes(
      tfl_converted_int16_model
    )
    interpreter_tool_converted = runtime.Interpreter.from_bytes(
      int8_converted_int16_model
    )

    num_steps = 10
    # Give the same (random) input to both interpreters to confirm that the outputs are similar.
    np.random.seed(42)
    for _ in range(0, num_steps):
      data_x = np.random.random((1, 28, 28)).astype("float32")

      tfl_converted_result = inference(interpreter_tfl_converted, data_x)[0]
      tool_converted_result = inference(interpreter_tool_converted, data_x)[0]

      max_diff = max(abs(tool_converted_result - tfl_converted_result))
      self.assertLess(
        max_diff, 1e-4
      )  # can not be the same since int8 model loses some range information


if __name__ == "__main__":
  unittest.main()
