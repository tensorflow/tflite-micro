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
"""
Wake-word model evaluation, with audio preprocessing using MicroInterpreter

Run:
bazel build tensorflow/lite/micro/examples/micro_speech:evaluate
bazel-bin/tensorflow/lite/micro/examples/micro_speech/evaluate
  --sample_path="path to 1 second audio sample in WAV format"
"""

from absl import app
from absl import flags
import numpy as np
from pathlib import Path

from tflite_micro.python.tflite_micro import runtime
from tensorflow.python.platform import resource_loader
import tensorflow as tf
from tflite_micro.tensorflow.lite.micro.examples.micro_speech import audio_preprocessor

_SAMPLE_PATH = flags.DEFINE_string(
    name='sample_path',
    default='',
    help='path for the audio sample to be predicted.',
)

_FEATURES_SHAPE = (49, 40)


def quantize_input_data(data, input_details):
  """quantize the input data using scale and zero point

  Args:
      data (np.array in float): input data for the interpreter
      input_details : output of get_input_details from the tflm interpreter.

  Returns:
    np.ndarray: quantized data as int8 dtype
  """
  # Get input quantization parameters
  data_type = input_details['dtype']
  input_quantization_parameters = input_details['quantization_parameters']
  input_scale, input_zero_point = input_quantization_parameters['scales'][
      0], input_quantization_parameters['zero_points'][0]
  # quantize the input data
  data = data / input_scale + input_zero_point
  return data.astype(data_type)


def dequantize_output_data(data: np.ndarray,
                           output_details: dict) -> np.ndarray:
  """Dequantize the model output

  Args:
      data: integer data to be dequantized
      output_details: TFLM interpreter model output details

  Returns:
      np.ndarray: dequantized data as float32 dtype
  """
  output_quantization_parameters = output_details['quantization_parameters']
  output_scale = output_quantization_parameters['scales'][0]
  output_zero_point = output_quantization_parameters['zero_points'][0]
  # Caveat: tflm_output_quant need to be converted to float to avoid integer
  # overflow during dequantization
  # e.g., (tflm_output_quant -output_zero_point) and
  # (tflm_output_quant + (-output_zero_point))
  # can produce different results (int8 calculation)
  return output_scale * (data.astype(np.float32) - output_zero_point)


def predict(interpreter: runtime.Interpreter,
            features: np.ndarray) -> np.ndarray:
  """
  Use TFLM interpreter to predict wake-word from audio sample features

  Args:
      interpreter: TFLM python interpreter instance
      features: wake-word model feature data, with shape _FEATURES_SHAPE

  Returns:
      np.ndarray: predicted probability (softmax) for each model category
  """

  input_details = interpreter.get_input_details(0)
  # Quantize the input if the model is quantized
  # and our features are np.float32
  if input_details['dtype'] != np.float32 and features.dtype == np.float32:
    features = quantize_input_data(features, input_details)
  flattened_features = features.flatten().reshape([1, -1])
  interpreter.set_input(flattened_features, 0)
  interpreter.invoke()
  tflm_output = interpreter.get_output(0)

  output_details = interpreter.get_output_details(0)
  if output_details['dtype'] == np.float32:
    return tflm_output[0].astype(np.float32)
  # Dequantize the output for quantized model
  return dequantize_output_data(tflm_output[0], output_details)


def generate_features(
    audio_pp: audio_preprocessor.AudioPreprocessor) -> np.ndarray:
  """
  Generate audio sample features

  Args:
      audio_pp: AudioPreprocessor instance

  Returns:
      np.ndarray: generated audio sample features with shape _FEATURES_SHAPE
  """
  if audio_pp.params.use_float_output:
    dtype = np.float32
  else:
    dtype = np.int8
  features = np.zeros(_FEATURES_SHAPE, dtype=dtype)
  start_index = 0
  window_size = int(audio_pp.params.window_size_ms *
                    audio_pp.params.sample_rate / 1000)
  window_stride = int(audio_pp.params.window_stride_ms *
                      audio_pp.params.sample_rate / 1000)
  samples = audio_pp.samples[0]
  frame_number = 0
  end_index = start_index + window_size

  # reset audio preprocessor noise estimates
  audio_pp.reset_tflm()

  while end_index <= len(samples):
    frame_tensor: tf.Tensor = tf.convert_to_tensor(
        samples[start_index:end_index])
    frame_tensor = tf.reshape(frame_tensor, [1, -1])
    feature_tensor = audio_pp.generate_feature_using_tflm(frame_tensor)
    features[frame_number] = feature_tensor.numpy()
    start_index += window_stride
    end_index += window_stride
    frame_number += 1

  return features


def get_category_names() -> list[str]:
  """
  Get the list of model output category names

  Returns:
      list[str]: model output category names
  """
  return ['silence', 'unknown', 'yes', 'no']


def _main(_):
  sample_path = Path(_SAMPLE_PATH.value)
  assert sample_path.exists() and sample_path.is_file(), \
      'Audio sample file does not exist. Please check the path.'
  model_prefix_path = resource_loader.get_path_to_datafile('models')
  model_path = Path(model_prefix_path, 'micro_speech_quantized.tflite')

  feature_params = audio_preprocessor.FeatureParams()
  audio_pp = audio_preprocessor.AudioPreprocessor(feature_params)
  audio_pp.load_samples(sample_path)
  features = generate_features(audio_pp)

  tflm_interpreter = runtime.Interpreter.from_file(model_path)

  frame_number = 0
  test_features = np.zeros(_FEATURES_SHAPE, dtype=np.int8)
  for feature in features:
    test_features[frame_number] = feature
    category_probabilities = predict(tflm_interpreter, test_features)
    category_probabilities_str = '['
    for i in range(len(category_probabilities)):
      if i > 0:
        category_probabilities_str += ', '
      category_probabilities_str += f'{category_probabilities[i]:.4f}'
    category_probabilities_str += ']'
    print(f'Frame #{frame_number}: {category_probabilities_str}')
    frame_number += 1

  category_probabilities = predict(tflm_interpreter, features)
  predicted_category = np.argmax(category_probabilities)
  category_names = get_category_names()
  print('Model predicts the audio sample as'
        f' <{category_names[predicted_category]}>'
        f' with probability {category_probabilities[predicted_category]:.2f}')


if __name__ == '__main__':
  app.run(_main)
