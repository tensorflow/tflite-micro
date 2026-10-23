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
  """Returns the smallest supported power of 2 FFT length larger than or equal

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
    raise ValueError(
      "Invalid fft_length. Must be between %d and %d."
      % (_MIN_FFT_LENGTH, _MAX_FFT_LENGTH)
    )
  return fft_length, fft_bits


def _validate_fft_length(fft_length):
  if not (
    (_MIN_FFT_LENGTH <= fft_length <= _MAX_FFT_LENGTH) and (fft_length % 2 == 0)
  ):
    raise ValueError(
      "Invalid fft_length. Must be an even number between %d and %d."
      % (_MIN_FFT_LENGTH, _MAX_FFT_LENGTH)
    )


def rfft(input_tensor, fft_length, name="signal_rfft"):
  """Computes the 1-D discrete Fourier transform of a real-valued signal.

  Computes the RFFT over the innermost dimension of `input_tensor`. Since the
  DFT of a real signal is Hermitian-symmetric, RFFT returns only the
  `(fft_length // 2) + 1` unique complex components: the zero-frequency (DC)
  term followed by the `fft_length // 2` positive-frequency terms. If
  `fft_length` is larger than the innermost dimension of `input_tensor`, the
  dimension is zero-padded.

  Args:
    input_tensor: A `Tensor` of type `float32`, `int16`, or `int32` with rank
      >= 1.
    fft_length: Even `int` in `[64, 2048]` specifying the FFT length. Shorter
      inputs are zero-padded to `fft_length`.
    name: Optional name for the operation.

  Returns:
    A `Tensor` of the same dtype as `input_tensor` containing
    `((fft_length // 2) + 1) * 2` interleaved real/imaginary elements along its
    innermost dimension. For integer inputs (`int16`, `int32`), the output is
    scaled by `1 / fft_length` to avoid overflow; for `float32`, it is unscaled.

  Raises:
    ValueError: If `fft_length` is not an even integer in `[64, 2048]`.
  """
  _validate_fft_length(fft_length)
  with tf.name_scope(name) as name:
    input_tensor = tf.convert_to_tensor(input_tensor)
    return gen_fft_ops.signal_rfft(
      input_tensor, fft_length=fft_length, name=name
    )


def irfft(input_tensor, fft_length, name="signal_irfft"):
  """Computes the inverse 1-D discrete Fourier transform of a real signal.

  Computes the inverse RFFT over the innermost dimension of `input_tensor`,
  which is assumed to contain the `(fft_length // 2) + 1` unique complex
  components produced by `rfft` (represented as `((fft_length // 2) + 1) * 2`
  interleaved real/imaginary elements).

  Args:
    input_tensor: A `Tensor` of type `float32`, `int16`, or `int32` containing
      `((fft_length // 2) + 1) * 2` interleaved spectral elements along its
      innermost dimension.
    fft_length: Even `int` in `[64, 2048]` specifying the inverse FFT length.
    name: Optional name for the operation.

  Returns:
    A `Tensor` of the same dtype as `input_tensor` containing `fft_length`
    time-domain elements along its innermost dimension.

  Raises:
    ValueError: If `fft_length` is not an even integer in `[64, 2048]`.
  """
  _validate_fft_length(fft_length)
  with tf.name_scope(name) as name:
    input_tensor = tf.convert_to_tensor(input_tensor)
    return gen_fft_ops.signal_irfft(
      input_tensor, fft_length=fft_length, name=name
    )


def fft_auto_scale(input_tensor, name="signal_fft_auto_scale"):
  """Left-shifts `input_tensor` to maximize amplitude without clipping.

  Returns both the scaled signal and the number of left-shift bits applied for
  downstream compensation. Useful for maximizing precision in 16-bit integer FFT
  pipelines.

  Args:
    input_tensor: A 1-D `int16` time-domain input signal.
    name: Optional name for the operation.

  Returns:
    A tuple `(output, scale_bits)` where `output` is the auto-scaled 1-D `int16`
    signal and `scale_bits` is a scalar `int32` `Tensor` giving the number of
    left shifts applied.

  Raises:
    ValueError: If `input_tensor` does not have rank 1.
  """
  with tf.name_scope(name) as name:
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int16)
    dim_list = input_tensor.shape.as_list()
    if len(dim_list) != 1:
      raise ValueError("Input tensor must have a rank of 1")
    return gen_fft_ops.signal_fft_auto_scale(input_tensor, name=name)


tf.no_gradient("signal_rfft")
tf.no_gradient("signal_irfft")
tf.no_gradient("signal_fft_auto_scale")
