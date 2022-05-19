# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import os

from tflite_micro.tensorflow.lite.micro.tools.python_interpreter.src import interpreter_wrapper_pybind


class Interpreter(object):

  def __init__(self, model_data, arena_size):
    self._interpreter = interpreter_wrapper_pybind.InterpreterWrapper(
        model_data, arena_size)

  @classmethod
  def from_file(self, model_path, arena_size=100000):
    if model_path is None or not os.path.isfile(model_path):
      raise ValueError("Invalid model file path")

    with open(model_path, "rb") as f:
      model_data = f.read()

    return Interpreter(model_data, arena_size)

  @classmethod
  def from_bytes(self, model_data, arena_size=100000):
    if model_data is None:
      raise ValueError("Model must not be None")
    return Interpreter(model_data, arena_size)

  def invoke(self):
    self._interpreter.Invoke()

  def set_input(self, input_data, index):
    if input_data is None:
      raise ValueError("Input data must not be None")
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    self._interpreter.SetInputTensor(input_data, index)

  def get_output(self, index):
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    return self._interpreter.GetOutputTensor(index)
