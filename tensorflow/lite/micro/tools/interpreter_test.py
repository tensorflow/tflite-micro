# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf
from absl import app
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

from tflite_micro.tensorflow.lite.micro.tools.pybind_wrapper import interpreter_wrapper_pybind


def main(_):
  # help(interpreter_wrapper_pybind)

  filepath = '/home/psho/tflite-micro/tensorflow/lite/micro/examples/hello_world/hello_world.tflite'
  with open(filepath, "rb") as f:
    data = f.read()

  # TFLM interpreter
  interpreter_wrapper = interpreter_wrapper_pybind.InterpreterWrapper(data)
  interpreter_wrapper.AllocateTensors()

  # TFLite interpreter
  tflite_interpreter = tf.lite.Interpreter(model_path=filepath)
  tflite_interpreter.allocate_tensors()
  tflite_output_details = tflite_interpreter.get_output_details()[0]
  tflite_input_details = tflite_interpreter.get_input_details()[0]

  num_steps = 1000
  output_data = np.empty(num_steps)
  tflite_output_data = np.empty(num_steps)
  diff = np.empty(num_steps)
  for i in range(0, num_steps):
    input_data = np.full((1, 1), i, dtype=np.int8)
    interpreter_wrapper.SetInputTensor(input_data, 0)
    interpreter_wrapper.Invoke()
    output = interpreter_wrapper.GetOutputTensor()
    output_data[i] = output

    tflite_interpreter.set_tensor(tflite_input_details['index'], input_data)
    tflite_interpreter.invoke()
    tflite_output = tflite_interpreter.get_tensor(
        tflite_output_details['index'])
    tflite_output_data[i] = tflite_output

    diff[i] = tflite_output - output

  # print(output_data)
  # print(tflite_output_data)
  print(diff)


if __name__ == '__main__':
  app.run(main)
