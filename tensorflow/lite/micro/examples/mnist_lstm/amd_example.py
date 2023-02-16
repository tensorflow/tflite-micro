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
# =============================================================================
"""Example for AMD folks to requantize int8 model to int16 and test accuacy

Run:
bazel build tensorflow/lite/micro/examples/mnist_lstm:amd_example
bazel-bin/tensorflow/lite/micro/examples/mnist_lstm/amd_example 
"""
from absl import app
import numpy as np

from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime
from tflite_micro.tensorflow.lite.micro.examples.mnist_lstm.evaluate import predict
from tflite_micro.tensorflow.lite.micro.tools.requantize_flatbuffer import Requantizer


def main(_):
  # Paths for the model
  float_model_path = "/tmp/AMD_no_reverb_float.tflite"
  int8_model_path = "/tmp/AMD_no_reverb_int8.tflite"
  # Initialize the interpreter
  tflm_interpreter_float = tflm_runtime.Interpreter.from_file(float_model_path)
  tflm_interpreter_int8 = tflm_runtime.Interpreter.from_file(int8_model_path)

  # Requantize the int8 model to int16 using the requantizer
  # Alternative: using shell script to requantize the int8 model
  # bazel-bin/tensorflow/lite/micro/tools/requantize_flatbuffer --int8_model_path=/tmp/AMD_no_reverb_int8.tflite --save_path=/tmp/requantized_amd_int16.tflite
  requantizer = Requantizer.from_file(int8_model_path)
  requantizer.requantize_8to16()
  int16_model = requantizer.model_bytearray()

  tflm_interpreter_int16 = tflm_runtime.Interpreter.from_bytes(int16_model)

  diff_reductions = []
  average_diffs_int8 = []
  np.random.seed(42)
  for _ in range(100):
    # Range deduced from the int8 model:
    # Max = 0.1354544758796692 * (127 + 128) ~ 34
    # Min = 0.1354544758796692 * (-128 + 128) = 0
    data = np.random.uniform(0, 34, (1, 1, 241)).astype(np.float32)

    float_result = predict(tflm_interpreter_float, data)
    int8_result = predict(tflm_interpreter_int8, data)
    int16_result = predict(tflm_interpreter_int16, data)

    int8_diff = abs(float_result - int8_result)
    int16_diff = abs(float_result - int16_result)

    average_diffs_int8.append(int8_diff.mean())
    # int8_diff - int16_diff: positive means error reduction
    diff_reductions.append(int8_diff.mean() - int16_diff.mean())

  print(f"Average diff reduction: {np.mean(diff_reductions)}")
  print(
      f"Average diff reduction rate: {np.mean(diff_reductions)/np.mean(average_diffs_int8)}"
  )


if __name__ == "__main__":
  app.run(main)
