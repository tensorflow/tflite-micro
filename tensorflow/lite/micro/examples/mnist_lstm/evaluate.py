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
"""LSTM model evaluation for MNIST recognition

Run:
bazel build tensorflow/lite/micro/examples/mnist_lstm:evaluate
bazel-bin/tensorflow/lite/micro/examples/mnist_lstm/evaluate
--model_path=".tflite file path" --img_path="MNIST image path"

"""
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image

from tflite_micro.python.tflite_micro import runtime

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "/tmp/lstm_trained_model/lstm.tflite",
                    "the trained model path.")
flags.DEFINE_string("img_path", "/tmp/samples/sample0.jpg",
                    "path for the image to be predicted.")


def read_img(img_path):
  """Read MNIST image

  Args:
      img_path (str): path to a MNIST image

  Returns:
      np.array : image in the correct np.array format
  """
  image = Image.open(img_path)
  data = np.asarray(image, dtype=np.float32)
  if data.shape not in [(28, 28), (28, 28, 1)]:
    raise ValueError(
        "Invalid input image shape (MNIST image should have shape 28*28 or 28*28*1)"
    )
  # Normalize the image if necessary
  if data.max() > 1:
    data = data / 255.0
  # Model inference requires batch size one
  data = data.reshape((1, 28, 28))
  return data


def quantize_input_data(data, input_details):
  """quantize the input data using scale and zero point

  Args:
      data (np.array in float): input data for the interpreter
      input_details : output of get_input_details from the tflm interpreter.
  """
  # Get input quantization parameters
  data_type = input_details["dtype"]
  input_quantization_parameters = input_details["quantization_parameters"]
  input_scale, input_zero_point = input_quantization_parameters["scales"][
      0], input_quantization_parameters["zero_points"][0]
  # quantize the input data
  data = data / input_scale + input_zero_point
  return data.astype(data_type)


def dequantize_output_data(data, output_details):
  """Dequantize the data

  Args:
      data (int8 or int16): integer data that need to be dequantized
      output_details : output of get_output_details from the tflm interpreter.
  """
  output_quantization_parameters = output_details["quantization_parameters"]
  output_scale, output_zero_point = output_quantization_parameters["scales"][
      0], output_quantization_parameters["zero_points"][0]
  # Caveat: tflm_output_quant need to be converted to float to avoid integer overflow during dequantization
  # e.g., (tflm_output_quant -output_zero_point) and (tflm_output_quant + (-output_zero_point))
  # can produce different results (int8 calculation)
  return output_scale * (data.astype("float") - output_zero_point)


def tflm_predict(tflm_interpreter, data):
  """Predict using the tflm interpreter

  Args:
      tflm_interpreter (Interpreter): TFLM interpreter
      data (np.array): data that need to be predicted

  Returns:
      prediction (np.array): predicted results from the model using TFLM interpreter
  """
  tflm_interpreter.set_input(data, 0)
  tflm_interpreter.invoke()
  return tflm_interpreter.get_output(0)


def predict(interpreter, data):
  """Use TFLM interpreter to predict a MNIST image

  Args:
      interpreter (runtime.Interpreter): the TFLM python interpreter
      data (np.array): data to be predicted

  Returns:
      np.array : predicted probability (integer version if quantized) for each class (digit 0-9)
  """

  input_details = interpreter.get_input_details(0)
  # Quantize the input if the model is quantized
  if input_details["dtype"] != np.float32:
    data = quantize_input_data(data, input_details)
  interpreter.set_input(data, 0)
  interpreter.invoke()
  tflm_output = interpreter.get_output(0)

  # LSTM is stateful, reset the state after the usage since each image is independent
  interpreter.reset()
  output_details = interpreter.get_output_details(0)
  if output_details["dtype"] == np.float32:
    return tflm_output[0].astype("float")
  # Dequantize the output for quantized model
  return dequantize_output_data(tflm_output[0], output_details)


def predict_image(interpreter, image_path):
  """Use TFLM interpreter to predict a MNIST image

  Args:
      interpreter (runtime.Interpreter): the TFLM python interpreter
      image_path (str): path for the image that need to be tested

  Returns:
      np.array : predicted probability (integer version if quantized) for each class (digit 0-9)
  """
  data = read_img(image_path)
  return predict(interpreter, data)


def main(_):
  if not os.path.exists(FLAGS.model_path):
    raise ValueError(
        "Model file does not exist. Please check the .tflite model path.")
  if not os.path.exists(FLAGS.img_path):
    raise ValueError("Image file does not exist. Please check the image path.")

  tflm_interpreter = runtime.Interpreter.from_file(FLAGS.model_path)
  category_probabilities = predict_image(tflm_interpreter, FLAGS.img_path)
  predicted_category = np.argmax(category_probabilities)
  logging.info("Model predicts the image as %i with probability %.2f",
               predicted_category, category_probabilities[predicted_category])


if __name__ == "__main__":
  app.run(main)
