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

import os
import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.micro.examples.hello_world import evaluate

PREFIX_PATH = resource_loader.get_path_to_datafile('')


class HelloWorldFloatModelTest(test_util.TensorFlowTestCase):
  model_path = os.path.join(PREFIX_PATH, 'models/hello_world_float.tflite')
  input_shape = (1, 1)
  output_shape = (1, 1)
  tflm_interpreter = runtime.Interpreter.from_file(model_path)

  def test_compare_with_tflite(self):
    x_values = evaluate.generate_random_float_input()

    tflm_y_predictions = evaluate.get_tflm_prediction(self.model_path,
                                                      x_values)

    tflite_y_predictions = evaluate.get_tflite_prediction(
        self.model_path, x_values)

    self.assertAllEqual(tflm_y_predictions, tflite_y_predictions)


class HelloWorldQuantModelTest(test_util.TensorFlowTestCase):
  model_path = os.path.join(PREFIX_PATH, 'models/hello_world_int8.tflite')
  input_shape = (1, 1)
  output_shape = (1, 1)
  tflm_interpreter = runtime.Interpreter.from_file(model_path)

  def test_compare_with_tflite(self):
    x_values = evaluate.generate_random_int8_input()

    tflm_y_predictions = evaluate.get_tflm_prediction(self.model_path,
                                                      x_values)

    tflite_y_predictions = evaluate.get_tflite_prediction(
        self.model_path, x_values)

    self.assertAllEqual(tflm_y_predictions, tflite_y_predictions)


if __name__ == '__main__':
  test.main()
