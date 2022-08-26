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
"""
LSTM model evaluation for MNIST recognition
"""
import argparse
import numpy as np
import tensorflow as tf
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime


def get_test_data():
  """Get the total MNIST test data

  Returns:
      tuple: a tuple of test data and label
  """
  _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_test = x_test / 255.  # normalize pixel values to 0-1
  x_test = x_test.astype(np.float32)
  return (x_test, y_test)


def random_sample_data(tot_data, labels):
  """Select a random data sample for testing

  Args:
      tot_data (numpy.array): the total MNIST testing dataset in shape: [batch,28,28]
      labels (numpy.array): the corresponding label

  Returns:
      tuple: selected data and label
  """
  idx = np.random.randint(0, len(tot_data))
  sample_data = tot_data[idx:idx+1,:,:]
  sample_label = labels[idx]
  return (sample_data, sample_label)


def main(model_path, num_test):
  """Run MNIST LSTM model inference using TFLM interpreter

  Args:
      model_path (str): path to the .tflite model
      num_test (int) : number of test to run
  """
  tflm_interpreter = tflm_runtime.Interpreter.from_file(model_path)
  x_test, y_test = get_test_data()
  for _ in range(num_test):
    data, label = random_sample_data(x_test, y_test)
    tflm_interpreter.set_input(data, 0)
    tflm_interpreter.invoke()
    tflm_output = tflm_interpreter.get_output(0)[0]
    pred = np.argmax(tflm_output)
    print(
        f'Model predict {pred} with probability {tflm_output[pred]:.2f}, label : {label}'
    )

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run LSTM model using TFLM Python Interpreter')

  parser.add_argument('--model_path',
                      required=True,
                      type=str,
                      help='path to the .tflite model')

  parser.add_argument('--rounds',
                      type=int,
                      default=5,
                      help='number of inference to run')
  args = parser.parse_args()
  main(args.model_path, args.rounds)
