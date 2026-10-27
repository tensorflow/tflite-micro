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
"""Use delay op in python."""

import tensorflow as tf
from tflite_micro.python.tflite_micro.signal.utils import util

gen_delay_op = util.load_custom_op('delay_op.so')


# TODO(b/286250473): change back name after name clash resolved
def delay(input_tensor, delay_length, name="signal_delay"):
  """Delays the innermost dimension of `input_tensor` by `delay_length` samples.

  For example, assuming an input signal of 10 samples:
    `[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]`
  Calling `delay` configured with `delay_length=3` produces:
    `[0, 0, 0, 1, 2, 3, 4, 5, 6, 7]`
  To retrieve the remainder of the input signal, call `delay` again with zeros
  as input (`[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`) to get:
    `[8, 9, 0, 0, 0, 0, 0, 0, 0, 0]`

  Args:
    input_tensor: A multidimensional `int16` input signal (rank >= 1).
    delay_length: Non-negative `int` number of samples to delay the innermost
      dimension.
    name: Optional name for the operation.

  Returns:
    An `int16` `Tensor` of the same shape as `input_tensor`, with the innermost
    dimension delayed by `delay_length` samples.
  """
  with tf.name_scope(name) as name:
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int16)
    return gen_delay_op.signal_delay(
      input_tensor, delay_length=delay_length, name=name
    )


tf.no_gradient("signal_delay")
