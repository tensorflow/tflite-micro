# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Python utility script to provide unit test model data."""

import os
import sys


def create_conv_model_from_tf():
  """Creates a basic Keras model and converts to tflite."""
  import numpy as np
  import tensorflow as tf

  np.random.seed(0)
  input_shape = (16, 16, 1)

  model = tf.keras.models.Sequential()
  model.add(
    tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=input_shape)
  )
  model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
  model.add(tf.keras.layers.MaxPooling2D(2))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(10))
  model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
  )

  data_x = np.random.rand(12, 16, 16, 1)
  data_y = np.random.randint(2, size=(12, 10))
  model.fit(data_x, data_y, epochs=5)

  def representative_dataset_gen():
    np.random.seed(0)
    for _ in range(12):
      yield [np.random.rand(16, 16).reshape(1, 16, 16, 1).astype(np.float32)]

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8
  converter.representative_dataset = representative_dataset_gen
  converter._experimental_disable_per_channel_quantization_for_dense_layers = (  # pylint: disable=protected-access
    True
  )
  return converter.convert()


def generate_conv_model(
  write_to_file=True, filename="/tmp/tf_micro_conv_test_model.int8.tflite"
):
  """Loads the pregenerated conv int8 model and optionally writes it to file."""
  model_path = os.path.join(os.path.dirname(__file__), "conv_test_model.tflite")
  with open(model_path, "rb") as f:
    tflite_model = f.read()

  if write_to_file:
    with open(filename, "wb") as f:
      f.write(tflite_model)

  return tflite_model


def main(argv):
  del argv  # Unused for now
  generate_conv_model()


if __name__ == "__main__":
  main(sys.argv)
