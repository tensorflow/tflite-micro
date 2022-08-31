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
from PIL import Image
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime
from tflite_micro.tensorflow.lite.micro.examples.mnist_lstm import train
from tflite_micro.tensorflow.lite.micro.examples.mnist_lstm import evaluate


class LSTMModelTest(test_util.TensorFlowTestCase):

  model_dir = "/tmp/lstm_trained_model/"
  model_path = model_dir + 'lstm.tflite'
  if not os.path.exists(model_path):
    train.train_save_model(model_dir, epochs=3)

  input_shape = (1, 28, 28)
  output_shape = (1, 10)

  def testCompareWithTFLite(self):
    tflite_interpreter = tf.lite.Interpreter(model_path=self.model_path)
    tflite_interpreter.allocate_tensors()
    tflite_output_details = tflite_interpreter.get_output_details()[0]
    tflite_input_details = tflite_interpreter.get_input_details()[0]

    num_steps = 100
    for _ in range(0, num_steps):
      # Create random input
      data_x = np.random.random(self.input_shape)
      data_x = data_x.astype('float32')
      tflite_interpreter.set_tensor(tflite_input_details["index"], data_x)
      tflite_interpreter.invoke()
      tflite_output = tflite_interpreter.get_tensor(
          tflite_output_details["index"])

      # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
      # the states.
      # Clean up internal states.
      tflite_interpreter.reset_all_variables()

      # Run inference on TFLM
      # TODO(b/244330968): reset interpreter states instead of initialzing everytime
      tflm_interpreter = tflm_runtime.Interpreter.from_file(self.model_path)
      tflm_interpreter.set_input(data_x, 0)
      tflm_interpreter.invoke()
      tflm_output = tflm_interpreter.get_output(0)

      # Check that TFLM has correct output
      self.assertDTypeEqual(tflm_output, np.float32)
      self.assertEqual(tflm_output.shape, self.output_shape)
      self.assertAllLessEqual((tflite_output - tflm_output), 1e-5)

  def testModelAccuracy(self):
    sample_img_dir = "tensorflow/lite/micro/examples/mnist_lstm/samples/"
    img_idx = [0, 2, 4, 6, 8]
    img_lables = [7, 1, 4, 4, 5]

    for img_id, label in zip(img_idx, img_lables):
      # TODO(b/244330968): reset interpreter states instead of initialzing everytime
      tflm_interpreter = tflm_runtime.Interpreter.from_file(self.model_path)
      img_path = sample_img_dir + f"sample{img_id}.png"

      # Run inference on the sample image
      probs = evaluate.predict_image(tflm_interpreter, img_path)

      # Check the prediction result
      pred = np.argmax(probs)
      self.assertEqual(pred, label)


if __name__ == "__main__":
  test.main()
