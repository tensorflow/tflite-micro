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
"""Python package for TFLM Python Interpreter"""

import os

from tflite_micro.tensorflow.lite.micro.python.interpreter.src import interpreter_wrapper_pybind


class Interpreter(object):

  def __init__(self, model_data, custom_op_registerers, arena_size):
    if model_data is None:
      raise ValueError("Model must not be None")

    if not isinstance(custom_op_registerers, list) or not all(
        isinstance(s, str) for s in custom_op_registerers):
      raise ValueError("Custom ops registerers must be a list of strings")

    # This is a heuristic to ensure that the arena is sufficiently sized.
    if arena_size is None:
      arena_size = len(model_data) * 10

    self._interpreter = interpreter_wrapper_pybind.InterpreterWrapper(
        model_data, custom_op_registerers, arena_size)

  @classmethod
  def from_file(self, model_path, custom_op_registerers=[], arena_size=None):
    """Instantiates a TFLM interpreter from a model .tflite filepath.

    Args:
      model_path: Filepath to the .tflite model
      custom_op_registerers: List of strings, each of which is the name of a
        custom OP registerer
      arena_size: Tensor arena size in bytes. If unused, tensor arena size will
        default to 10 times the model size.

    Returns:
      An Interpreter instance
    """
    if model_path is None or not os.path.isfile(model_path):
      raise ValueError("Invalid model file path")

    with open(model_path, "rb") as f:
      model_data = f.read()

    return Interpreter(model_data, custom_op_registerers, arena_size)

  @classmethod
  def from_bytes(self, model_data, custom_op_registerers=[], arena_size=None):
    """Instantiates a TFLM interpreter from a model in byte array.

    Args:
      model_data: Model in byte array format
      custom_op_registerers: List of strings, each of which is the name of a
        custom OP registerer
      arena_size: Tensor arena size in bytes. If unused, tensor arena size will
        default to 10 times the model size.

    Returns:
      An Interpreter instance
    """

    return Interpreter(model_data, custom_op_registerers, arena_size)

  def invoke(self):
    """Invoke the TFLM interpreter to run an inference.

    This should be called after `set_input()`.

    Returns:
      Status code of the C++ invoke function. A RuntimeError will be raised as
      well upon any error.
    """
    return self._interpreter.Invoke()

  def set_input(self, input_data, index):
    """Set input data into input tensor.

    This should be called before `invoke()`.

    Args:
      input_data: Input data in numpy array format. The numpy array format is
        chosen to be consistent with TFLite interpreter.
      index: An integer between 0 and the number of input tensors (exclusive)
        consistent with the order defined in the list of inputs in the .tflite
        model
    """
    if input_data is None:
      raise ValueError("Input data must not be None")
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    self._interpreter.SetInputTensor(input_data, index)

  def get_output(self, index):
    """Get data from output tensor.

    The output data correspond to the most recent `invoke()`.

    Args:
      index: An integer between 0 and the number of output tensors (exclusive)
        consistent with the order defined in the list of outputs in the .tflite
        model

    Returns:
      Output data in numpy array format. The numpy array format is chosen to
      be consistent with TFLite interpreter.
    """
    if index is None or index < 0:
      raise ValueError("Index must be a non-negative integer")

    return self._interpreter.GetOutputTensor(index)
