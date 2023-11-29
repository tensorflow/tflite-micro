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
# ==============================================================================
"""Use overlap add op in python."""

import tensorflow as tf
from tflite_micro.python.tflite_micro.signal.utils import util

gen_delay_op = util.load_custom_op('delay_op.so')


def _delay_wrapper(delay_fn, default_name):
  """Wrapper around gen_delay_op.delay*."""

  def _delay(input_tensor, delay_length, name=default_name):
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int16)
      return delay_fn(input_tensor, delay_length=delay_length, name=name)

  return _delay


# TODO(b/286250473): change back name after name clash resolved
delay = _delay_wrapper(gen_delay_op.signal_delay, "signal_delay")

tf.no_gradient("signal_delay")
