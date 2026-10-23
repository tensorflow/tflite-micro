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
"""Use PCAN op in python."""

import tensorflow as tf
from tflite_micro.python.tflite_micro.signal.utils import util
from tflite_micro.python.tflite_micro.signal.utils import (
  wide_dynamic_func_lut_wrapper,
)

gen_pcan_op = util.load_custom_op("pcan_op.so")

PCAN_SNR_BITS = 12


# TODO(b/286250473): change back name after name clash resolved
def pcan(
  input_tensor,
  noise_estimate,
  strength,
  offset,
  gain_bits,
  smoothing_bits,
  input_correction_bits,
  name="signal_pcan",
):
  """Applies per-channel amplitude-normalized (PCAN) automatic gain control.

  Compresses the dynamic range of mel filter bank channels by dividing by a
  power of the per-channel `noise_estimate` using a fixed-point lookup table
  generated from `strength`, `offset`, `smoothing_bits`,
  `input_correction_bits`, and `gain_bits`.

  Args:
    input_tensor: A 1-D `uint32` array of mel-spectrum filter bank outputs.
    noise_estimate: A 1-D `uint32` array of per-channel noise estimates.
    strength: Floating-point PCAN normalization exponent (strength).
    offset: Floating-point PCAN positive offset added to the noise estimate.
    gain_bits: Number of fractional bits in the lookup-table gain values.
    smoothing_bits: Fractional bits associated with `noise_estimate`.
    input_correction_bits: Bit-shift correction applied prior to PCAN.
    name: Optional name for the operation.

  Returns:
    A 1-D `uint32` `Tensor` with the same shape as `noise_estimate` containing
    gain-controlled filter bank channels.

  Raises:
    ValueError: If the computed SNR shift (`gain_bits - input_correction_bits -
      PCAN_SNR_BITS`) is less than 1, or if `input_tensor` or `noise_estimate`
      does not have rank 1.
  """
  with tf.name_scope(name) as scope:
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint32)
    noise_estimate = tf.convert_to_tensor(noise_estimate, dtype=tf.uint32)

    input_bits = smoothing_bits - input_correction_bits
    snr_shift = gain_bits - input_correction_bits - PCAN_SNR_BITS
    if snr_shift < 1:
      raise ValueError("SNR shift must be non-negative: %d" % snr_shift)

    lut = wide_dynamic_func_lut_wrapper.wide_dynamic_func_lut(
      strength, offset, input_bits, gain_bits
    )

    lut_tensor = tf.convert_to_tensor(lut, dtype=tf.int16)

    dim_list = input_tensor.shape.as_list()
    if len(dim_list) != 1:
      raise ValueError("Input tensor must have a rank of 1")
    dim_list = noise_estimate.shape.as_list()
    if len(dim_list) != 1:
      raise ValueError("Noise estimate must have a rank of 1")

    snr_shift = 6
    return gen_pcan_op.signal_pcan(
      input_tensor,
      noise_estimate,
      lut_tensor,
      snr_shift=snr_shift,
      name=scope,
    )


tf.no_gradient("pcan")
