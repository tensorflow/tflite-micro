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


def invoke_tflm_interpreter(input_shape, interpreter, x_quantized, input_index,
                            output_index):
  input_data = np.reshape(x_quantized, input_shape)
  interpreter.set_input(input_data, input_index)
  interpreter.invoke()
  y_quantized = np.reshape(interpreter.get_output(output_index), -1)[0]
  return y_quantized


def invoke_tflite_interpreter(input_shape, interpreter, x_quantized,
                              input_index, output_index):
  input_data = np.reshape(x_quantized, input_shape)
  interpreter.set_tensor(input_index, input_data)
  interpreter.invoke()
  tflite_output = interpreter.get_tensor(output_index)
  y_quantized = np.reshape(tflite_output, -1)[0]
  return y_quantized


# Generate a list of 1000 random floats in the range of 0 to 2*pi.
def generate_random_input(sample_count=1000):
  # Generate a uniformly distributed set of random numbers in the range from
  # 0 to 2π, which covers a complete sine wave oscillation
  x_values = np.random.uniform(
      low=0, high=2 * np.pi, size=sample_count).astype(np.float32)
  # Shuffle the values to guarantee they're not in order
  np.random.shuffle(x_values)
  return x_values


# Get the metadata like scales and zero_points from the interpreter input/output
# details.
def get_metadata(interpreter_io_details):
  quantize_params = interpreter_io_details.get('quantization_parameters')
  scale = quantize_params.get('scales')
  zero_point = quantize_params.get('zero_points')
  return scale, zero_point


# Invoke the tflm interpreter with x_values in the range of [0, 2*PI] and
# returns the prediction of the interpreter.
def get_tflm_prediction(tflm_interpreter, x_values):
  input_details = tflm_interpreter.get_input_details(0)
  input_scale, input_zero_point = get_metadata(input_details)

  output_details = tflm_interpreter.get_output_details(0)
  output_scale, output_zero_point = get_metadata(output_details)

  input_shape = np.array(input_details.get('shape'))

  y_predictions = np.empty(x_values.size, dtype=np.float32)
  i = 0
  for x_value in x_values:
    # Quantize the input from floating-point to integer
    x_quantized = np.int8((x_value / input_scale) + input_zero_point)
    y_quantized = invoke_tflm_interpreter(
        input_shape,
        tflm_interpreter,
        x_quantized,
        input_index=0,
        output_index=0)
    y_predictions[i] = float((y_quantized - output_zero_point) * output_scale)
    i += 1
    # print("x : {} y_pred : {} y_expected : {}".format(x_value, y_pred, y_expected))
  return y_predictions


# Invoke the tflite interpreter with x_values in the range of [0, 2*PI] and
# returns the prediction of the interpreter.
def get_tflite_prediction(tflite_interpreter, x_values):
  input_details = tflite_interpreter.get_input_details()[0]
  output_details = tflite_interpreter.get_output_details()[0]
  input_shape = np.array(input_details.get('shape'))

  input_scale, input_zero_point = get_metadata(input_details)
  output_scale, output_zero_point = get_metadata(output_details)

  y_predictions = np.empty(x_values.size, dtype=np.float32)
  i = 0
  for x_value in x_values:
    x_quantized = np.int8((x_value / input_scale) + input_zero_point)
    y_quantized = invoke_tflite_interpreter(input_shape, tflite_interpreter,
                                            x_quantized, input_details['index'],
                                            output_details['index'])
    y_predictions[i] = float((y_quantized - output_zero_point) * output_scale)
    i += 1
  return y_predictions


def main(_):
  if _MODEL_PATH.value is None:
    raise ValueError('Invalid tflm model file path')

  hello_world_model_path = os.path.join(PREFIX_PATH, _MODEL_PATH.value)
  # Create the tflm interpreter
  interpreter = tflm_runtime.Interpreter.from_file(hello_world_model_path)

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