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
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.tools import requantize_flatbuffer
from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.tools import flatbuffer_utils


# TODO(b/248061370): replace the keras model creation process with flatbuffer manipulation to speed up test
def create_simple_fc_model():
  '''Create a simple model with two fully connected(fc) layers'''
  model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="output")
  ])
  fixed_input = tf.keras.layers.Input(shape=[28, 28],
                                      batch_size=1,
                                      dtype=model.inputs[0].dtype,
                                      name="fixed_input")
  fixed_output = model(fixed_input)
  return tf.keras.models.Model(fixed_input, fixed_output)


def representative_dataset_gen(num_samples=100):
  np.random.seed(42)  #Seed the random number generator
  for _ in range(num_samples):
    yield [np.random.random((1, 28, 28)).astype(np.float32)]


def convert_tfl_converter(keras_model,
                          representative_dataset_gen,
                          int16=False):
  '''Convert and quantize the keras model using the standard tflite converter'''
  converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  if int16:
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.
        EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
  converter.representative_dataset = representative_dataset_gen
  # TODO(b/324385802): Support per-channel quantization for FullyConnected.
  converter._experimental_disable_per_channel_quantization_for_dense_layers = True
  converter._experimental_disable_per_channel = True
  return converter.convert()


def convert_8to16_requantizer(keras_model, representative_dataset_gen):
  '''Convert and quantize the keras model using the int8 to int16 conversion tool'''
  # Convert to int8 first
  int8_model = convert_tfl_converter(keras_model,
                                     representative_dataset_gen,
                                     int16=False)
  int8_model = flatbuffer_utils.convert_bytearray_to_object(int8_model)
  # Use the tool to convert to int16
  requantizer = requantize_flatbuffer.Requantizer(int8_model)
  requantizer.requantize_8to16()
  return flatbuffer_utils.convert_object_to_bytearray(requantizer.model)


class SimpleFCModelTest(test_util.TensorFlowTestCase):

  def testCompareWithStandardConversion(self):

    def inference(tflm_interpreter, data_x):
      tflm_interpreter.set_input(data_x, 0)
      tflm_interpreter.invoke()
      return tflm_interpreter.get_output(0)

    keras_model = create_simple_fc_model(
    )  # int16 fc is supported in tflite converter
    tfl_converted_int16_model = convert_tfl_converter(
        keras_model, representative_dataset_gen, int16=True)
    int8_converted_int16_model = convert_8to16_requantizer(
        keras_model, representative_dataset_gen)

    interpreter_tfl_converted = runtime.Interpreter.from_bytes(
        tfl_converted_int16_model)
    interpreter_tool_converted = runtime.Interpreter.from_bytes(
        int8_converted_int16_model)

    num_steps = 10
    # Give the same (random) input to both interpreters to confirm that the outputs are similar.
    for _ in range(0, num_steps):
      data_x = np.random.random((1, 28, 28)).astype("float32")

      tfl_converted_result = inference(interpreter_tfl_converted, data_x)[0]
      tool_converted_result = inference(interpreter_tool_converted, data_x)[0]

      max_diff = max(abs(tool_converted_result - tfl_converted_result))
      self.assertLess(
          max_diff, 1e-4
      )  # can not be the same since int8 model loses some range information


if __name__ == "__main__":
  test.main()
