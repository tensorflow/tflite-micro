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

# A simple test to check whether the tflite_micro package works after it is
# installed.

# To test from the perspective of a package user, use import paths to locations
# in the Python installation environment rather than to locations in the tflm
# source tree.
from tflite_micro import runtime

import numpy as np
import pkg_resources
import sys


def passed():
  # Create an interpreter with a sine model
  model = pkg_resources.resource_filename(__name__, "sine_float.tflite")
  interpreter = runtime.Interpreter.from_file(model)
  OUTPUT_INDEX = 0
  INPUT_INDEX = 0
  input_shape = interpreter.get_input_details(INPUT_INDEX).get("shape")

  # The interpreter infers sin(x)
  def infer(x):
    tensor = np.array(x, np.float32).reshape(input_shape)
    interpreter.set_input(tensor, INPUT_INDEX)
    interpreter.invoke()
    return interpreter.get_output(OUTPUT_INDEX).squeeze()

  # Check a few inferred values against a numerical computation
  PI = 3.14
  inputs = (0.0, PI / 2, PI, 3 * PI / 2, 2 * PI)
  outputs = [infer(x) for x in inputs]
  goldens = np.sin(inputs)

  return np.allclose(outputs, goldens, atol=0.05)


if __name__ == "__main__":
  sys.exit(0 if passed() else 1)
