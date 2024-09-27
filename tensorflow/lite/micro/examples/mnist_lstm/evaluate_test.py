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
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.micro.examples.mnist_lstm import evaluate
from tflite_micro.tensorflow.lite.micro.tools import requantize_flatbuffer

PREFIX_PATH = resource_loader.get_path_to_datafile("")


class LSTMFloatModelTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.model_path = os.path.join(PREFIX_PATH, "trained_lstm.tflite")
    self.input_shape = (1, 28, 28)
    self.output_shape = (1, 10)
    self.tflm_interpreter = runtime.Interpreter.from_file(self.model_path)
    np.random.seed(42)  #Seed the random number generator

  def testInputErrHandling(self):
    wrong_size_image_path = os.path.join(PREFIX_PATH, "samples/resized9.png")
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             "Invalid input image shape"):
      evaluate.predict_image(self.tflm_interpreter, wrong_size_image_path)

  def testCompareWithTFLite(self):
    tflite_interpreter = tf.lite.Interpreter(
        model_path=self.model_path,
        experimental_op_resolver_type=\
        tf.lite.experimental.OpResolverType.BUILTIN_REF)
    tflite_interpreter.allocate_tensors()
    tflite_output_details = tflite_interpreter.get_output_details()[0]
    tflite_input_details = tflite_interpreter.get_input_details()[0]

    num_steps = 100
    for _ in range(0, num_steps):
      # Clear the internal states of the TfLite and TFLM interpreters so that we can call invoke multiple times (LSTM is stateful).
      tflite_interpreter.reset_all_variables()
      self.tflm_interpreter.reset()

      # Give the same (random) input to both interpreters can confirm that the output is identical.
      data_x = np.random.random(self.input_shape)
      data_x = data_x.astype("float32")

      # Run inference on TFLite
      tflite_interpreter.set_tensor(tflite_input_details["index"], data_x)
      tflite_interpreter.invoke()
      tflite_output = tflite_interpreter.get_tensor(
          tflite_output_details["index"])

      # Run inference on TFLM
      tflm_output = evaluate.tflm_predict(self.tflm_interpreter, data_x)

      # Check that TFLM has correct output
      self.assertDTypeEqual(tflm_output, np.float32)
      self.assertEqual(tflm_output.shape, self.output_shape)
      self.assertAllLess((tflite_output - tflm_output), 1e-5)

  def testModelAccuracy(self):
    # Test prediction accuracy on digits 0-9 using sample images
    for label in range(10):
      image_path = os.path.join(PREFIX_PATH, f"samples/sample{label}.png")
      # Run inference on the sample image
      # Note that the TFLM state is reset inside the predict_image function.
      category_probabilities = evaluate.predict_image(self.tflm_interpreter,
                                                      image_path)
      # Check the prediction result
      predicted_category = np.argmax(category_probabilities)
      self.assertEqual(predicted_category, label)


class LSTMInt8ModelTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.int8_model_path = os.path.join(PREFIX_PATH,
                                        "trained_lstm_int8.tflite")
    self.input_shape = (1, 28, 28)
    self.output_shape = (1, 10)
    self.tflm_interpreter_quant = runtime.Interpreter.from_file(
        self.int8_model_path)
    np.random.seed(42)  #Seed the random number generator

  def testQuantOutputs(self):
    # Get input/output information of the quantized model
    input_details = self.tflm_interpreter_quant.get_input_details(0)
    output_details = self.tflm_interpreter_quant.get_output_details(0)

    # Create a float model for results comparison
    float_model_path = os.path.join(PREFIX_PATH, "trained_lstm.tflite")
    tflm_interpreter_float = runtime.Interpreter.from_file(float_model_path)

    num_test = 10
    for _ in range(num_test):
      # Clear the internal states of the TfLite and TFLM interpreters so that we can call invoke multiple times (LSTM is stateful).
      self.tflm_interpreter_quant.reset()
      tflm_interpreter_float.reset()

      data_x = np.random.random(self.input_shape)
      data_x = data_x.astype("float32")

      # Run float inference on TFLM
      tflm_output_float = evaluate.tflm_predict(tflm_interpreter_float, data_x)

      # Quantized the input data into int8
      data_x_quant = evaluate.quantize_input_data(data_x, input_details)

      # Run integer inference on the quantilzed TFLM model
      tflm_output_quant = evaluate.tflm_predict(self.tflm_interpreter_quant,
                                                data_x_quant)
      # Check shape and type
      self.assertDTypeEqual(tflm_output_quant, np.int8)
      self.assertEqual(tflm_output_quant.shape, self.output_shape)

      # Convert the integer output back to float for comparison
      tflm_output_quant_float = evaluate.dequantize_output_data(
          tflm_output_quant, output_details)
      # Make sure the difference is within the error margin
      self.assertAllLess(abs(tflm_output_float - tflm_output_quant_float),
                         1e-2)

  def testQuantModelAccuracy(self):
    for label in range(10):
      image_path = os.path.join(PREFIX_PATH, f"samples/sample{label}.png")
      # Run integer inference (quantized) on the sample image
      # Note that the TFLM state is reset inside the predict_image function.
      category_probabilities_quant = evaluate.predict_image(
          self.tflm_interpreter_quant, image_path)
      # Check the prediction result
      predicted_category = np.argmax(category_probabilities_quant)
      # Check the prediction
      self.assertEqual(predicted_category, label)


class LSTMInt16ModelTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # Convert the int8 model to int16
    self.int8_model_path = os.path.join(PREFIX_PATH,
                                        "trained_lstm_int8.tflite")
    self.requantizer = requantize_flatbuffer.Requantizer.from_file(
        self.int8_model_path)
    self.requantizer.requantize_8to16()
    self.int16_model = self.requantizer.model_bytearray()
    self.input_shape = (1, 28, 28)
    self.output_shape = (1, 10)
    self.tflm_interpreter_quant = runtime.Interpreter.from_bytes(
        self.int16_model)
    np.random.seed(42)  #Seed the random number generator

  def testQuantOutputs(self):
    # Get input/output information
    input_details = self.tflm_interpreter_quant.get_input_details(0)
    output_details = self.tflm_interpreter_quant.get_output_details(0)

    # Create a float model for results comparison
    float_model_path = os.path.join(PREFIX_PATH, "trained_lstm.tflite")
    tflm_interpreter_float = runtime.Interpreter.from_file(float_model_path)

    num_test = 10
    for _ in range(num_test):
      # Clear the internal states of the TfLite and TFLM interpreters so that we can call invoke multiple times (LSTM is stateful).
      self.tflm_interpreter_quant.reset()
      tflm_interpreter_float.reset()

      data_x = np.random.random(self.input_shape)
      data_x = data_x.astype("float32")

      # Run float inference on TFLM
      tflm_output_float = evaluate.tflm_predict(tflm_interpreter_float, data_x)

      # Quantized the input data into int8
      data_x_quant = evaluate.quantize_input_data(data_x, input_details)

      # Run integer inference on the quantilzed TFLM model
      tflm_output_quant = evaluate.tflm_predict(self.tflm_interpreter_quant,
                                                data_x_quant)
      # Check shape and type
      self.assertDTypeEqual(tflm_output_quant, np.int16)
      self.assertEqual(tflm_output_quant.shape, self.output_shape)

      # Convert the integer output back to float for comparison
      tflm_output_quant_float = evaluate.dequantize_output_data(
          tflm_output_quant, output_details)
      # Make sure the difference is within the error margin
      self.assertAllLess(abs(tflm_output_float - tflm_output_quant_float),
                         1e-3)

  def testQuantModelAccuracy(self):
    for label in range(10):
      image_path = os.path.join(PREFIX_PATH, f"samples/sample{label}.png")
      # Run integer inference (quantized) on the sample image
      # Note that the TFLM state is reset inside the predict_image function.
      category_probabilities_quant = evaluate.predict_image(
          self.tflm_interpreter_quant, image_path)
      # Check the prediction result
      predicted_category = np.argmax(category_probabilities_quant)
      # Check the prediction
      self.assertEqual(predicted_category, label)


if __name__ == "__main__":
  test.main()
