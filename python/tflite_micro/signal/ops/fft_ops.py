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
"""Use FFT ops in python."""

import math
import tensorflow as tf
from tflite_micro.python.tflite_micro.signal.utils import util

gen_fft_ops = util.load_custom_op('fft_ops.so')

_MIN_FFT_LENGTH = 64
_MAX_FFT_LENGTH = 2048


def get_pow2_fft_length(input_length):
  """Returns the smallest suuported power of 2 FFT length larger than or equal

  to the input_length.

  Only returns FFT lengths that are powers of 2 within the range
  [_MIN_FFT_LENGTH, _MAX_FFT_LENGTH].

  Args:
    input_length: Length of input time domain signal.

  Returns:
    A pair: the smallest length and its log2 (number of bits)

  Raises:
    ValueError: The FFT length needed is not supported
  """
  fft_bits = math.ceil(math.log2(input_length))
  fft_length = pow(2, fft_bits)
  if not _MIN_FFT_LENGTH <= fft_length <= _MAX_FFT_LENGTH:
    raise ValueError("Invalid fft_length. Must be between %d and %d." %
                     (_MIN_FFT_LENGTH, _MAX_FFT_LENGTH))
  return fft_length, fft_bits


def _fft_wrapper(fft_fn, default_name):
  """Wrapper around gen_fft_ops.*rfft*."""

  def _fft(input_tensor, fft_length, name=default_name):
    if not ((_MIN_FFT_LENGTH <= fft_length <= _MAX_FFT_LENGTH) and
            (fft_length % 2 == 0)):
      raise ValueError(
          "Invalid fft_length. Must be an even number between %d and %d." %
          (_MIN_FFT_LENGTH, _MAX_FFT_LENGTH))
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor)
      return fft_fn(input_tensor, fft_length=fft_length, name=name)

  return _fft


def _fft_auto_scale_wrapper(fft_auto_scale_fn, default_name):
  """Wrapper around gen_fft_ops.fft_auto_scale*."""

  def _fft_auto_scale(input_tensor, name=default_name):
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int16)
      dim_list = input_tensor.shape.as_list()
      if len(dim_list) != 1:
        raise ValueError("Input tensor must have a rank of 1")
      return fft_auto_scale_fn(input_tensor, name=name)

  return _fft_auto_scale


rfft = _fft_wrapper(gen_fft_ops.signal_rfft, "signal_rfft")
irfft = _fft_wrapper(gen_fft_ops.signal_irfft, "signal_irfft")
fft_auto_scale = _fft_auto_scale_wrapper(gen_fft_ops.signal_fft_auto_scale,
                                         "signal_fft_auto_scale")
tf.no_gradient("signal_rfft")
tf.no_gradient("signal_irfft")
tf.no_gradient("signal_fft_auto_scale")
