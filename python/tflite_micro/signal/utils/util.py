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
# ==============================================================================
"""Python utility functions."""
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro import runtime


# TODO(b/286889497): find better name and place for this function.
def get_tflm_interpreter(concrete_function, trackable_obj):
  """Initialize a TFLite interpreter with a concerte function.

  Args:
    concrete_function: A concrete function

  Returns:
    TFLite interpreter object
  """
  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [concrete_function], trackable_obj)
  converter.allow_custom_ops = True
  tflite_model = converter.convert()

  return runtime.Interpreter.from_bytes(tflite_model, arena_size=500000)


def load_custom_op(name):
  return load_library.load_op_library(
      resource_loader.get_path_to_datafile('../ops/_' + name))
