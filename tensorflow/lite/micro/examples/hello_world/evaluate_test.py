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
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime
from tflite_micro.tensorflow.lite.micro.examples.hello_world import evaluate

PREFIX_PATH = resource_loader.get_path_to_datafile('')


class HelloWorldQuantModelTest(test_util.TensorFlowTestCase):
  model_path = os.path.join(PREFIX_PATH, 'models/hello_world_float.tflite')
  input_shape = (1, 1)
  output_shape = (1, 1)
  # Create the tflm interpreter
  tflm_interpreter = tflm_runtime.Interpreter.from_file(model_path)

  # Get the metadata like scales and zero_points from the interpreter input/output
  # details.
  def get_quantization_params(self, interpreter_io_details):
    quantize_params = interpreter_io_details.get('quantization_parameters')
    scale = quantize_params.get('scales')
    zero_point = quantize_params.get('zero_points')
    return scale, zero_point

  def test_input(self):
    input_details = self.tflm_interpreter.get_input_details(0)
    input_scale, input_zero_point = self.get_quantization_params(input_details)

    self.assertAllEqual(input_details['shape'], self.input_shape)
    self.assertEqual(input_details['dtype'], np.float32)
    self.assertEqual(len(input_scale), 0)
    self.assertEqual(
        input_details['quantization_parameters']['quantized_dimension'], 0)
    self.assertEqual(input_scale.dtype, np.float32)
    self.assertEqual(input_zero_point.dtype, np.int32)

  def test_output(self):
    output_details = self.tflm_interpreter.get_output_details(0)
    output_scale, output_zero_point = self.get_quantization_params(
        output_details)
    self.assertAllEqual(output_details['shape'], self.output_shape)
    self.assertEqual(output_details['dtype'], np.float32)
    self.assertEqual(len(output_scale), 0)
    self.assertEqual(
        output_details['quantization_parameters']['quantized_dimension'], 0)
    self.assertEqual(output_scale.dtype, np.float32)
    self.assertEqual(output_zero_point.dtype, np.int32)

  def test_interpreter_prediction(self):
    x_value = np.float32(0.0)
    # Calculate the corresponding sine values
    y_true = np.sin(x_value).astype(np.float32)

    input_shape = np.array(
        self.tflm_interpreter.get_input_details(0).get('shape'))

    y_pred = evaluate.invoke_tflm_interpreter(
        input_shape,
        self.tflm_interpreter,
        x_value,
        input_index=0,
        output_index=0,
    )

    epsilon = 0.05
    self.assertNear(
        y_true,
        y_pred,
        epsilon,
        'hello_world model prediction is not close enough to numpy.sin value',
    )

  def test_compare_with_tflite(self):
    x_values = evaluate.generate_random_input()

    tflm_y_predictions = evaluate.get_tflm_prediction(self.model_path,
                                                      x_values)

    tflite_y_predictions = evaluate.get_tflite_prediction(
        self.model_path, x_values)

    self.assertAllEqual(tflm_y_predictions, tflite_y_predictions)


if __name__ == '__main__':
  test.main()
