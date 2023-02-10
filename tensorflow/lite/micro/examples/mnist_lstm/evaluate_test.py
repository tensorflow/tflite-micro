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
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime
from tflite_micro.tensorflow.lite.micro.examples.mnist_lstm import evaluate

PREFIX_PATH = resource_loader.get_path_to_datafile("")


class LSTMFloatModelTest(test_util.TensorFlowTestCase):

  model_path = os.path.join(PREFIX_PATH, "trained_lstm.tflite")
  input_shape = (1, 28, 28)
  output_shape = (1, 10)

  tflm_interpreter = tflm_runtime.Interpreter.from_file(model_path)
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
      self.tflm_interpreter.set_input(data_x, 0)
      self.tflm_interpreter.invoke()
      tflm_output = self.tflm_interpreter.get_output(0)

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


class LSTMQuantModelTest(test_util.TensorFlowTestCase):

  quant_model_path = os.path.join(PREFIX_PATH, "trained_lstm_int8.tflite")
  input_shape = (1, 28, 28)
  output_shape = (1, 10)

  tflm_interpreter_quant = tflm_runtime.Interpreter.from_file(quant_model_path)
  np.random.seed(42)  #Seed the random number generator

  def testQuantOutputs(self):
    # Get input/output quantization parameters
    input_quantization_parameters = self.tflm_interpreter_quant.get_input_details(
        0)["quantization_parameters"]
    output_quantization_parameters = self.tflm_interpreter_quant.get_output_details(
        0)["quantization_parameters"]
    input_scale, input_zero_point = input_quantization_parameters["scales"][
        0], input_quantization_parameters["zero_points"][0]
    output_scale, output_zero_point = output_quantization_parameters["scales"][
        0], output_quantization_parameters["zero_points"][0]
    # Create a float model for results comparison
    float_model_path = os.path.join(PREFIX_PATH, "trained_lstm.tflite")
    tflm_interpreter_float = tflm_runtime.Interpreter.from_file(
        float_model_path)

    num_test = 100
    for _ in range(num_test):
      # Clear the internal states of the TfLite and TFLM interpreters so that we can call invoke multiple times (LSTM is stateful).
      self.tflm_interpreter_quant.reset()
      tflm_interpreter_float.reset()

      data_x = np.random.random(self.input_shape)
      data_x = data_x.astype("float32")

      # Run float inference on TFLM
      tflm_interpreter_float.set_input(data_x, 0)
      tflm_interpreter_float.invoke()
      tflm_output_float = tflm_interpreter_float.get_output(0)

      # Quantized the input data into int8
      data_x_quant = data_x / input_scale + input_zero_point
      data_x_quant = data_x_quant.astype("int8")

      # Run integer inference on the quantilzed TFLM model
      self.tflm_interpreter_quant.set_input(data_x_quant, 0)
      self.tflm_interpreter_quant.invoke()
      tflm_output_quant = self.tflm_interpreter_quant.get_output(0)
      # Check shape and type
      self.assertDTypeEqual(tflm_output_quant, np.int8)
      self.assertEqual(tflm_output_quant.shape, self.output_shape)

      # Convert the integer output back to float for comparison
      # Caveat: tflm_output_quant need to be converted to float to avoid integer overflow during dequantization
      # e.g., (tflm_output_quant -output_zero_point) and (tflm_output_quant + (-output_zero_point))
      # can produce different results (int8 calculation)
      tflm_output_quant_float = output_scale * (
          tflm_output_quant.astype("float") - output_zero_point)
      # Make sure the difference is within the error margin
      self.assertAllLess(abs(tflm_output_float - tflm_output_quant_float),
                         1e-2)

  def testQuantModelAccuracy(self):
    for label in range(10):
      image_path = os.path.join(PREFIX_PATH, f"samples/sample{label}.png")
      # Run integer inference (quantized) on the sample image
      # Note that the TFLM state is reset inside the predict_image function.
      category_probabilities_quant = evaluate.predict_image(
          self.tflm_interpreter_quant, image_path, quantized=True)
      # Check the prediction result
      predicted_category = np.argmax(category_probabilities_quant)
      # Check the prediction
      self.assertEqual(predicted_category, label)


if __name__ == "__main__":
  test.main()
