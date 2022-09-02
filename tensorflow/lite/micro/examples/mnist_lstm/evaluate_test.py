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

from absl import logging
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime
from tflite_micro.tensorflow.lite.micro.examples.mnist_lstm import train
from tflite_micro.tensorflow.lite.micro.examples.mnist_lstm import evaluate


class LSTMModelTest(test_util.TensorFlowTestCase):

  model_dir = "/tmp/lstm_trained_model/"
  model_path = model_dir + 'lstm.tflite'
  if not os.path.exists(model_path):
    logging.info("No trained LSTM model. Training in Keras now (3 epoches)")
    train.train_save_model(model_dir, epochs=3)

  input_shape = (1, 28, 28)
  output_shape = (1, 10)

  tflm_interpreter = tflm_runtime.Interpreter.from_file(model_path)

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
      tflm_output = evaluate.predict(self.tflm_interpreter, data_x)
      self.tflm_interpreter.reset()

      # Check that TFLM has correct output
      self.assertDTypeEqual(tflm_output, np.float32)
      self.assertEqual(tflm_output.shape, self.output_shape)
      self.assertAllLess((tflite_output - tflm_output), 1e-5)

  def testModelAccuracy(self):
    sample_img_dir = "tensorflow/lite/micro/examples/mnist_lstm/samples/"
    img_lables = [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]

    for i, label in enumerate(img_lables):
      img_path = sample_img_dir + f"sample{i}.png"

      # Run inference on the sample image
      probs = evaluate.predict_image(self.tflm_interpreter, img_path)
      self.tflm_interpreter.reset()

      # Check the prediction result
      pred = np.argmax(probs)
      self.assertEqual(pred, label)


if __name__ == "__main__":
  test.main()
