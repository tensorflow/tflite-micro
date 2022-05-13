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

try:
  # Try to import from bazel build first
  from tflite_micro.tensorflow.lite.micro.tools.python_interpreter.src import interpreter_wrapper_pybind
except ImportError:
  # Resort to pip package
  import interpreter_wrapper_pybind

import sys

class Interpreter(object):
  def __init__(self, model_path, arena_size=10000):
    with open(model_path, "rb") as f:
      model_data = f.read()
    self._interpreter = interpreter_wrapper_pybind.InterpreterWrapper(model_data, arena_size)
    print(self._interpreter)
    sys.stdout.flush()

  def invoke(self):
    self._interpreter.Invoke()

  def set_input(self, input_data, index):
    self._interpreter.SetInputTensor(input_data, index)

  def get_output(self):
    return self._interpreter.GetOutputTensor()
