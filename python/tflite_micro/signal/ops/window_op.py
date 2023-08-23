# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Use window op in python."""

import numpy as np
import tensorflow as tf
from tflite_micro.python.tflite_micro.signal.utils import util

gen_window_op = util.load_custom_op('window_op.so')


def hann_window_weights(window_length, shift, dtype=np.int16):
  arg = np.pi * 2 / window_length
  index = np.arange(window_length)
  weights = (0.5 - (0.5 * np.cos(arg * (index + 0.5))))
  if dtype == np.int16:
    weights = np.round(weights * (2**shift))
  return weights.astype(dtype=dtype)


# We can calculate the result of sqrt(0.5-cos(x)/2) without sqrt as sin(x/2).
def square_root_hann_window_weights(window_length, shift, dtype=np.int16):
  arg_half = np.pi / window_length
  index = np.arange(window_length)
  weights = np.sin(arg_half * (index + 0.5))
  if dtype == np.int16:
    weights = np.round(weights * (2**shift))
  return weights.astype(dtype=dtype)


# In the so-called weighted overlap add (WOLA) method, a second window would
# be applied after the inverse FFT and prior to the final overlap-add to
# generate the output signal. This second window is known as a synthesis
# window and it is commonly chosen to be the same as the first window (which
# is applied before the FFT and is known as an analysis window). The pair
# of windows need to be normalized such that they together meet the constant
# WOLA (CWOLA) constraint. So if a signal goes through this procedure, it can
# be reconstructed with little distortion. For the square-root Hann window
# implemented above, the normalizing constant is given by
# sqrt((window_length / (2 * window_step)).
def square_root_hann_cwola_window_weights(window_length,
                                          window_step,
                                          shift,
                                          dtype=np.int16):
  arg_half = np.pi / window_length
  norm = np.sqrt(window_length / (2.0 * window_step))
  index = np.arange(window_length)
  weights = np.sin(arg_half * (index + 0.5)) / norm
  if dtype == np.int16:
    weights = np.round(weights * (2**shift))
  return weights.astype(dtype=dtype)


def _window_wrapper(window_fn, default_name):
  """Wrapper around gen_window_op.window*."""

  def _window(input_tensor, weight_tensor, shift, name=default_name):
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=np.int16)
      input_dim_list = input_tensor.shape.as_list()
      weight_tensor = tf.convert_to_tensor(weight_tensor)
      weight_dim_list = weight_tensor.shape.as_list()
      if input_dim_list[-1] != weight_dim_list[0]:
        raise ValueError("Innermost input dimension must match weights size")
      return window_fn(input_tensor, weight_tensor, shift=shift, name=name)

  return _window


# TODO(b/286250473): change back name to "window" after name clash resolved
window = _window_wrapper(gen_window_op.signal_window, "signal_window")

tf.no_gradient("signal_window")
