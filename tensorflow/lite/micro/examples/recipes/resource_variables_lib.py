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
#
"""Simple TF model creation using resource variables.

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


def get_tf_model(concrete_model=False):
  """Returns a TF custom accumulator model: by default a Keras model, otherwise concrete function model."""
  if concrete_model:
    return CompareAndAccumulate("CompareAndAccumulate")

  input_layer_int = tf.keras.layers.Input(
      shape=[100], dtype=tf.float32, name="accum_val"
  )
  input_layer_bool = tf.keras.layers.Input(
      shape=[1], dtype=tf.bool, name="accumulate_add"
  )
  accumulate_out = CompareAndAccumulateKerasLayer("CompareAndAccumulate")(
      input_layer_int, input_layer_bool
  )

  return tf.keras.models.Model(
      inputs=[input_layer_int, input_layer_bool], outputs=accumulate_out
  )


def convert_and_save_model(
    model, concrete_model=False, filename="/tmp/resource_var_model.tflite"
):
  """Converts a TF model into a .tflite model and saves to path given by filename.

  If model was created via concrete functions, use concrete_model,
  otherwise assumes a keras model.
  Args:
    model: TF Keras Model or TF.Module concrete function model
    concrete_model: if model is Keras or concrete
    filename: path to write the converted model to
  Returns:
    The converted tflite model
  """
  if concrete_model:
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],
                                                                model)
  else:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open(filename, "wb") as f:
    f.write(tflite_model)
  return tflite_model
