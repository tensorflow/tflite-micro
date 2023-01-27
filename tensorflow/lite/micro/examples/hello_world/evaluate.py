# pip install matplotlib -- this is required to plot the graph
import os
from absl import app
from absl import flags
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import resource_loader
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import \
  tflm_runtime

_MODEL_PATH = flags.DEFINE_string(
    'model_path', '', 'path to the tflite file that should be used.')

x_range = 2 * np.pi
inferences_per_cycle = 1000
PREFIX_PATH = resource_loader.get_path_to_datafile('')


def invoke_tflm_interpreter(input_scale, input_shape, input_zero_point,
                            interpreter, output_scale, output_zero_point,
                            x_value):
  x_quantized = np.int8((x_value / input_scale) + input_zero_point)
  input_data = np.reshape(x_quantized, input_shape)
  interpreter.set_input(input_data, 0)
  interpreter.invoke()
  y_quantized = np.reshape(interpreter.get_output(0), -1)[0]
  y_pred = float((y_quantized - output_zero_point) * output_scale)
  return y_pred


def invoke_tflite_interpreter(input_scale, input_shape, input_zero_point,
                              interpreter, output_scale, output_zero_point,
                              x_value, tflite_input_index, tflite_output_index):
  x_quantized = np.int8((x_value / input_scale) + input_zero_point)
  input_data = np.reshape(x_quantized, input_shape)
  interpreter.set_tensor(tflite_input_index, input_data)
  interpreter.invoke()
  tflite_output = interpreter.get_tensor(tflite_output_index)
  y_quantized = np.reshape(tflite_output, -1)[0]
  y_pred = float((y_quantized - output_zero_point) * output_scale)
  return y_pred


def generate_random_input():
  # Number of sample datapoints
  samples = 1000
  # Generate a uniformly distributed set of random numbers in the range from
  # 0 to 2π, which covers a complete sine wave oscillation
  x_values = np.random.uniform(
      low=0, high=2 * np.pi, size=samples).astype(np.float32)
  # Shuffle the values to guarantee they're not in order
  np.random.shuffle(x_values)
  return x_values


def get_metadata(details):
  quantize_params = details.get('quantization_parameters')
  scale = quantize_params.get('scales')
  zero_point = quantize_params.get('zero_points')
  return scale, zero_point


def get_tflm_prediction(interpreter, x_values):
  input_shape = np.array(interpreter.get_input_details(0).get('shape'))
  input_scale, input_zero_point = get_metadata(interpreter.get_input_details(0))
  output_scale, output_zero_point = get_metadata(
      interpreter.get_output_details(0))

  y_predictions = np.empty(x_values.size, dtype=np.float32)
  i = 0
  for x_value in x_values:
    # Quantize the input from floating-point to integer
    y_pred = invoke_tflm_interpreter(input_scale[0], input_shape,
                                     input_zero_point[0], interpreter,
                                     output_scale[0], output_zero_point[0],
                                     x_value)
    y_predictions[i] = y_pred
    i += 1
    # print("x : {} y_pred : {} y_expected : {}".format(x_value, y_pred, y_expected))
  return y_predictions


def get_tflite_prediction(tflite_interpreter, x_values):
  output_details = tflite_interpreter.get_output_details()[0]
  input_details = tflite_interpreter.get_input_details()[0]
  input_shape = np.array(input_details.get('shape'))

  input_scale, input_zero_point = get_metadata(input_details)
  output_scale, output_zero_point = get_metadata(output_details)

  y_predictions = np.empty(x_values.size, dtype=np.float32)
  i = 0
  for x_value in x_values:
    y_pred = invoke_tflite_interpreter(input_scale, input_shape,
                                       input_zero_point, tflite_interpreter,
                                       output_scale, output_zero_point, x_value,
                                       input_details['index'],
                                       output_details['index'])
    y_predictions[i] = y_pred
    i += 1
  return y_predictions


def main(_):
  if _MODEL_PATH.value is None:
    raise ValueError('Invalid tflm model file path')

  hello_world_model_path = os.path.join(PREFIX_PATH, _MODEL_PATH.value)
  # Create the tflm interpreter
  interpreter = tflm_runtime.Interpreter.from_file(
      hello_world_model_path, [], num_resource_variables=0)

  x_values = generate_random_input()

  # Calculate the corresponding sine values
  y_true_values = np.sin(x_values).astype(np.float32)

  y_predictions = get_tflm_prediction(interpreter, x_values)
  plt.plot(x_values, y_predictions, 'b.', label='TFLM Prediction')
  plt.plot(x_values, y_true_values, 'r.', label='Actual values')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  app.run(main)
