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
"""Creates a simple tflite model that adds two input tensor of size 1."""

from absl import app
import tensorflow as tf


def main(_):
  input_shape = (128, 128, 1)
  x1 = tf.keras.layers.Input(input_shape)
  x2 = tf.keras.layers.Input(input_shape)

  added = tf.keras.layers.Add()([x1, x2])
  model = tf.keras.models.Model(inputs=[x1, x2], outputs=added)

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # Enforce integer only quantization
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8

  # Fix random seed to keep the model reproducible.
  tf.random.set_seed(3)

  # Convert the model to the TensorFlow Lite format with quantization and
  # quantization requires a representative data set
  def representative_dataset():
    for i in range(500):
      yield ([
          tf.random.normal(input_shape, seed=i),
          tf.random.normal(input_shape, seed=i * 2)
      ])

  converter.representative_dataset = representative_dataset
  model_tflite = converter.convert()

  open("simple_add_model.tflite", "wb").write(model_tflite)


if __name__ == '__main__':
  app.run(main)
