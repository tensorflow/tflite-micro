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
"""Python utility script to generate unit test model data."""

# Steps to regenerate model test data:
# TODO(b/158011574): Do these steps in the script here instead of manually.
# 1.) Run this script
# 2.) Hexdump the model into a .h/.cc file:
#       xxd -i /tmp/tf_micro_conv_test_model.tflite > /tmp/temp.cc
# 3.) Copy/replace contents of temp.cc into desired header/source files (e.g.
#     test_conv_model.h/.cc

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Workaround for TensorFlow 2.19.0 keras lazy loader issue. If Keras 3.0 is
# installed, TensorFlow's lazy loader infinitely recurses when trying to
# determine which Keras version to use. This forces TensorFlow to use the
# one provided by package tf_keras. This must be set before importing
# `tensorflow`. See https://keras.io/getting_started/#tensorflow--keras-2-backwards-compatibility.
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from absl import app
import numpy as np
import tensorflow as tf


def generate_conv_model(write_to_file=True,
                        filename="/tmp/tf_micro_conv_test_model.int8.tflite"):
  """Creates a basic Keras model and converts to tflite.

  This model does not make any relevant classifications. It only exists to
  generate a model that is designed to run on embedded devices.
  """
  np.random.seed(0)
  input_shape = (16, 16, 1)

  model = tf.keras.models.Sequential()
  model.add(
      tf.keras.layers.Conv2D(16, 3, activation="relu",
                             input_shape=input_shape))
  model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
  model.add(tf.keras.layers.MaxPooling2D(2))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(10))
  model.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"])
  model.summary()

  # Test with random data
  data_x = np.random.rand(12, 16, 16, 1)
  data_y = np.random.randint(2, size=(12, 10))
  model.fit(data_x, data_y, epochs=5)

  def representative_dataset_gen():
    np.random.seed(0)
    for _ in range(12):
      yield [np.random.rand(16, 16).reshape(1, 16, 16, 1).astype(np.float32)]

  # Now convert to a TFLite model with full int8 quantization:
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8
  converter.representative_dataset = representative_dataset_gen
  # TODO(b/324385802): Disable per channel quantization in FC layers (currently
  # default behaviour) since it's not yet supported in TFLM.
  converter._experimental_disable_per_channel_quantization_for_dense_layers = (  # pylint: disable=protected-access
      True)

  tflite_model = converter.convert()
  if write_to_file:
    open(filename, "wb").write(tflite_model)

  return tflite_model


def main(argv):
  del argv  # Unused for now
  generate_conv_model()


if __name__ == "__main__":
  app.run(main)
