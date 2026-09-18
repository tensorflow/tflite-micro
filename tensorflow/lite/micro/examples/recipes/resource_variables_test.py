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
import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.examples.recipes import resource_variables_lib

from tflite_micro.python.tflite_micro import runtime as tflm_runtime


class ResourceVariablesTest(test_util.TensorFlowTestCase):

  # Tests the custom accumulator model. Input conditional is [True], and
  # accumulator value is array of 5.0. Given these inputs, we expect the output
  # (variable value), to be accumulated by 5.0 each invoke.
  def test_resource_variables_model(self):
    model_keras = resource_variables_lib.get_model_from_keras()
    tflm_interpreter = tflm_runtime.Interpreter.from_bytes(model_keras)

    tflm_interpreter.set_input([[True]], 0)
    tflm_interpreter.set_input([np.full((100,), 15.0, dtype=np.float32)], 1)
    tflm_interpreter.invoke()
    self.assertAllEqual(
        tflm_interpreter.get_output(0),
        np.full((1, 100), 15.0, dtype=np.float32),
    )

    tflm_interpreter.set_input([[False]], 0)
    tflm_interpreter.set_input([np.full((100,), 9.0, dtype=np.float32)], 1)
    tflm_interpreter.invoke()
    self.assertAllEqual(
        tflm_interpreter.get_output(0),
        np.full((1, 100), 6.0, dtype=np.float32),
    )

    # resets variables to initial value
    tflm_interpreter.reset()
    tflm_interpreter.set_input([[True]], 0)
    tflm_interpreter.set_input([np.full((100,), 5.0, dtype=np.float32)], 1)
    tflm_interpreter.invoke()
    self.assertAllEqual(
        tflm_interpreter.get_output(0),
        np.full((1, 100), 5.0, dtype=np.float32),
    )


if __name__ == "__main__":
  test.main()
