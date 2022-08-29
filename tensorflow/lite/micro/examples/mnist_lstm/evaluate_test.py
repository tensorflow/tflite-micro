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
# =============================================================================
import os
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime
from tflite_micro.tensorflow.lite.micro.examples.mnist_lstm import train


class LSTMModelTest(test_util.TensorFlowTestCase):
  model_dir = "/tmp/lstm_trained_model/"
  model_path = model_dir + 'lstm.tflite'
  if not os.path.exists(model_path):
    train.main(model_dir)

  input_shape = (1, 28, 28)
  output_shape = (1, 10)

  def testCompareWithTFLite(self):
    tflm_interpreter = tflm_runtime.Interpreter.from_file(self.model_path)

    tflite_interpreter = tf.lite.Interpreter(model_path=self.model_path)
    tflite_interpreter.allocate_tensors()
    tflite_output_details = tflite_interpreter.get_output_details()[0]
    tflite_input_details = tflite_interpreter.get_input_details()[0]

    num_steps = 100
    for i in range(0, num_steps):
      # Create random input
      data_x = np.random.random(self.input_shape)
      data_x = data_x.astype('float32')
      tflite_interpreter.set_tensor(tflite_input_details["index"], data_x)
      tflite_interpreter.invoke()
      tflite_output = tflite_interpreter.get_tensor(
          tflite_output_details["index"])

      # Run inference on TFLM
      tflm_interpreter.set_input(data_x, 0)
      tflm_interpreter.invoke()
      tflm_output = tflm_interpreter.get_output(0)

      # Check that TFLM output has correct metadata
      self.assertDTypeEqual(tflm_output, np.float32)
      self.assertEqual(tflm_output.shape, self.output_shape)
      self.assertAllLessEqual((tflite_output - tflm_output), 1)

if __name__ == "__main__":
  test.main()