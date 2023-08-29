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
"""Simple TF model creation using resource variables."""

import numpy as np
import tensorflow as tf


"""
Generates a simple TfLite model that adds 4 numbers.

Basic Usage:

  model = generate_model(False)

Usage where you want model written to file:

  file_path = "some file path"
  model = generate_model(True, file_path)
"""

class AddFourNumbers(tf.Module):
  @tf.function(
      input_signature=[
          tf.TensorSpec(shape=[1], dtype=tf.float32, name="a"),
          tf.TensorSpec(shape=[1], dtype=tf.float32, name="b"),
          tf.TensorSpec(shape=[1], dtype=tf.float32, name="c"),
          tf.TensorSpec(shape=[1], dtype=tf.float32, name="d"),
      ]
  )
  def __call__(self, a, b, c, d):
    return a + b + c + d


def get_model_from_concrete_function():
  """Accumulator model built via TF concrete functions."""
  model = AddFourNumbers("AddFourNumbers")
  concrete_func = model.__call__.get_concrete_function()
  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [concrete_func], model
  )
  return converter.convert()


def generate_model(write_file=True, filename="/tmp/add.tflite"):
  model = get_model_from_concrete_function()
  if write_file:
    with open(filename, "wb") as f:
      f.write(model)
  return model
