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
#
# Ueful links:
#  https://www.tensorflow.org/lite/models/convert/convert_models#convert_concrete_functions_
#  https://www.tensorflow.org/guide/function#creating_tfvariables
#  https://www.tensorflow.org/api_docs/python/tf/Variable

from absl import app
import numpy as np
import tensorflow as tf


class CustomAccumulate(tf.Module):
  def __init__(self, name):
    super().__init__(name=name)
    self._accum = tf.Variable(initial_value=tf.constant(np.full((100, ), 50.),
                                                        dtype=tf.float32),
                              trainable=False,
                              name="Accumulator",
                              dtype=tf.float32,
                              shape=[100])

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[100], dtype=tf.float32, name="input")
  ])
  def __call__(self, x):
    if x[0] < 10.:
      self._accum.assign_add(x)
    else:
      self._accum.assign_add(-x)

    return self._accum.read_value()


def main(_):
  model = CustomAccumulate("custom_accumulate")
  print("============================")
  print(model(tf.constant(np.full((100, ), 5.0), dtype=tf.float32)))
  print(model(tf.constant(np.full((100, ), 15.0), dtype=tf.float32)))
  print("============================")
  concrete_func = model.__call__.get_concrete_function()

  converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],
                                                              model)
  tflite_model = converter.convert()

  with open('/tmp/model.tflite', 'wb') as f:
    f.write(tflite_model)


if __name__ == "__main__":
  app.run(main)
