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
"""Basic Python test for the TFLM interpreter"""

# Steps to run this test:
#   bazel test tensorflow/lite/micro/tools/python_interpreter/tests:interpreter_test
#
# Steps to debug with gdb:
# 1. bazel build tensorflow/lite/micro/tools/python_interpreter/tests:interpreter_test
# 2. gdb python
# 3. (gdb) run bazel-out/k8-fastbuild/bin/tensorflow/lite/micro/tools/python_interpreter/tests/interpreter_test

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.testing import generate_test_models
from tflite_micro.tensorflow.lite.micro.tools.python_interpreter.src import tflm_runtime


class TFLiteComparisonTest(test_util.TensorFlowTestCase):

  def testCompareWithTFLite(self):
    model = generate_test_models.generate_conv_model(False)
    input_shape = (1, 16, 16, 1)
    output_shape = (1, 10)

    # TFLM interpreter
    tflm_interpreter = tflm_runtime.Interpreter.from_bytes(model)

    # TFLite interpreter
    tflite_interpreter = tf.lite.Interpreter(model_content=model)
    tflite_interpreter.allocate_tensors()
    tflite_output_details = tflite_interpreter.get_output_details()[0]
    tflite_input_details = tflite_interpreter.get_input_details()[0]

    num_steps = 1000
    for i in range(0, num_steps):
      # Create random input
      data_x = np.random.randint(-127, 127, input_shape, dtype=np.int8)

      # Run inference on TFLite
      tflite_interpreter.set_tensor(tflite_input_details['index'], data_x)
      tflite_interpreter.invoke()
      tflite_output = tflite_interpreter.get_tensor(
          tflite_output_details['index'])

      # Run inference on TFLM
      tflm_interpreter.set_input(data_x, 0)
      tflm_interpreter.invoke()
      tflm_output = tflm_interpreter.get_output(0)

      # Check that TFLM output has correct metadata
      self.assertDTypeEqual(tflm_output, np.int8)
      self.assertEqual(tflm_output.shape, output_shape)
      # Check that result differences are less than tolerance
      self.assertAllLessEqual((tflite_output - tflm_output), 1)


if __name__ == '__main__':
  test.main()
