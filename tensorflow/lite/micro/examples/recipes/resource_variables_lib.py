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
"""Simple TF model creation using resource variables.

Model is built either with basic TF functions (concrete function model), or via
Keras. The model simply mimics an accumulator (via the persistent memory / state
functionality of resource variables), taking in two inputs: 
1) A boolean for choosing addition/subtraction.
2) The value to add/subtract from the accumulator variable.


Useful links:
https://www.tensorflow.org/lite/models/convert/convert_models#convert_concrete_functions_
https://www.tensorflow.org/guide/function#creating_tfvariables
https://www.tensorflow.org/api_docs/python/tf/Variable
https://www.tensorflow.org/api_docs/python/tf/function
"""

import numpy as np
import tensorflow as tf


class CompareAndAccumulate(tf.Module):
  """Accumulates a given value to the resource variable array (initialized as 0.).

  Accumulates add/subtract based on second boolean input.
  """

  def __init__(self, name):
    super().__init__(name=name)
    self._accum = tf.Variable(
        initial_value=np.zeros((100,), dtype=np.float32),
        trainable=False,
        name="Accumulator",
        dtype=tf.float32,
        shape=[100],
    )

  @tf.function(
      input_signature=[
          tf.TensorSpec(shape=[100], dtype=tf.float32, name="accum_val"),
          tf.TensorSpec(shape=[1], dtype=tf.bool, name="accumulate_add"),
      ]
  )
  def __call__(self, accum_val, accumulate_add):
    if accumulate_add:
      self._accum.assign_add(accum_val)
    else:
      self._accum.assign_sub(accum_val)
    return self._accum.read_value()


class CompareAndAccumulateKerasLayer(tf.keras.layers.Layer):
  """Accumulates a given value to the resource variable array (initialized as 0.).

  Accumulates add/subtract based on second boolean input.
  """

  def __init__(self, name):
    super().__init__(name=name)
    self._accum = tf.Variable(
        initial_value=[np.zeros((100,), dtype=np.float32)],
        trainable=False,
        name="Accumulator",
        dtype=tf.float32,
        shape=(1, 100),
    )

  def call(self, accum_val, accumulate_add):
    @tf.function
    def condtional_accumulate(accum_val, accumulate_add):
      if accumulate_add:
        self._accum.assign_add(accum_val)
      else:
        self._accum.assign_sub(accum_val)
    condtional_accumulate(accum_val, accumulate_add)
    return self._accum.read_value()


def get_model_from_concrete_function():
  """Accumulator model built via TF concrete functions."""
  model = CompareAndAccumulate("CompareAndAccumulate")
  concrete_func = model.__call__.get_concrete_function()
  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [concrete_func], model
  )
  return converter.convert()


def get_model_from_keras():
  """Accumulator model built via Keras custom layer."""
  input_layer_int = tf.keras.layers.Input(
      shape=[100], dtype=tf.float32, name="accum_val"
  )
  input_layer_bool = tf.keras.layers.Input(
      shape=[1], dtype=tf.bool, name="accumulate_add"
  )
  accumulate_out = CompareAndAccumulateKerasLayer("CompareAndAccumulate")(
      input_layer_int, input_layer_bool
  )

  model = tf.keras.models.Model(
      inputs=[input_layer_int, input_layer_bool], outputs=accumulate_out
  )
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  return converter.convert()
