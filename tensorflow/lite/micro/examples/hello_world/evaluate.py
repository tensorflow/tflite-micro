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
import tensorflow as tf
from absl import app
from absl import flags
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro import runtime

_USE_TFLITE_INTERPRETER = flags.DEFINE_bool(
    'use_tflite',
    False,
    'Inference with the TF Lite interpreter instead of the TFLM interpreter',
)

_PREFIX_PATH = resource_loader.get_path_to_datafile('')


def invoke_tflm_interpreter(input_shape, interpreter, x_value, input_index,
                            output_index):
  input_data = np.reshape(x_value, input_shape)
  interpreter.set_input(input_data, input_index)
  interpreter.invoke()
  y_quantized = np.reshape(interpreter.get_output(output_index), -1)[0]
  return y_quantized


def invoke_tflite_interpreter(input_shape, interpreter, x_value, input_index,
                              output_index):
  input_data = np.reshape(x_value, input_shape)
  interpreter.set_tensor(input_index, input_data)
  interpreter.invoke()
  tflite_output = interpreter.get_tensor(output_index)
  y_quantized = np.reshape(tflite_output, -1)[0]
  return y_quantized


# Generate a list of 1000 random floats in the range of 0 to 2*pi.
def generate_random_int8_input(sample_count=1000):
  # Generate a uniformly distributed set of random numbers in the range from
  # 0 to 2π, which covers a complete sine wave oscillation
  np.random.seed(42)
  x_values = np.random.uniform(low=0, high=2 * np.pi,
                               size=sample_count).astype(np.int8)
  return x_values


# Generate a list of 1000 random floats in the range of 0 to 2*pi.
def generate_random_float_input(sample_count=1000):
  # Generate a uniformly distributed set of random numbers in the range from
  # 0 to 2π, which covers a complete sine wave oscillation
  np.random.seed(42)
  x_values = np.random.uniform(low=0, high=2 * np.pi,
                               size=sample_count).astype(np.float32)
  return x_values


# Invoke the tflm interpreter with x_values in the range of [0, 2*PI] and
# returns the prediction of the interpreter.
def get_tflm_prediction(model_path, x_values):
  # Create the tflm interpreter
  tflm_interpreter = runtime.Interpreter.from_file(model_path)

  input_shape = np.array(tflm_interpreter.get_input_details(0).get('shape'))

  y_predictions = np.empty(x_values.size, dtype=np.float32)

  for i, x_value in enumerate(x_values):
    y_predictions[i] = invoke_tflm_interpreter(input_shape,
                                               tflm_interpreter,
                                               x_value,
                                               input_index=0,
                                               output_index=0)
  return y_predictions


# Invoke the tflite interpreter with x_values in the range of [0, 2*PI] and
# returns the prediction of the interpreter.
def get_tflite_prediction(model_path, x_values):
  # TFLite interpreter
  tflite_interpreter = tf.lite.Interpreter(
      model_path=model_path,
      experimental_op_resolver_type=tf.lite.experimental.OpResolverType.
      BUILTIN_REF,
  )
  tflite_interpreter.allocate_tensors()

  input_details = tflite_interpreter.get_input_details()[0]
  output_details = tflite_interpreter.get_output_details()[0]
  input_shape = np.array(input_details.get('shape'))

  y_predictions = np.empty(x_values.size, dtype=np.float32)

  for i, x_value in enumerate(x_values):
    y_predictions[i] = invoke_tflite_interpreter(
        input_shape,
        tflite_interpreter,
        x_value,
        input_details['index'],
        output_details['index'],
    )
  return y_predictions


def main(_):
  model_path = os.path.join(_PREFIX_PATH, 'models/hello_world_float.tflite')

  x_values = generate_random_float_input()

  # Calculate the corresponding sine values
  y_true_values = np.sin(x_values).astype(np.float32)

  if _USE_TFLITE_INTERPRETER.value:
    y_predictions = get_tflite_prediction(model_path, x_values)
    plt.plot(x_values, y_predictions, 'b.', label='TFLite Prediction')
  else:
    y_predictions = get_tflm_prediction(model_path, x_values)
    plt.plot(x_values, y_predictions, 'b.', label='TFLM Prediction')

  plt.plot(x_values, y_true_values, 'r.', label='Actual values')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  app.run(main)
