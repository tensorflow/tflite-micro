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

import tensorflow as tf
from tflite_micro.python.tflite_micro.signal.utils import util
from tflite_micro.python.tflite_micro.signal.utils import wide_dynamic_func_lut_wrapper

gen_pcan_op = util.load_custom_op("pcan_op.so")

PCAN_SNR_BITS = 12


def _pcan_wrapper(pcan_fn, default_name):
  """Wrapper around gen_pcan.pcan*."""

  def _pcan(input_tensor,
            noise_estimate,
            strength,
            offset,
            gain_bits,
            smoothing_bits,
            input_correction_bits,
            name=default_name):
    with tf.name_scope(name) as scope:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint32)
      noise_estimate = tf.convert_to_tensor(noise_estimate, dtype=tf.uint32)

      input_bits = smoothing_bits - input_correction_bits
      snr_shift = gain_bits - input_correction_bits - PCAN_SNR_BITS
      if snr_shift < 1:
        raise ValueError("SNR shift must be non-negative: %d" % snr_shift)

      lut = wide_dynamic_func_lut_wrapper.wide_dynamic_func_lut(
          strength, offset, input_bits, gain_bits)

      lut_tensor = tf.convert_to_tensor(lut, dtype=tf.int16)

      dim_list = input_tensor.shape.as_list()
      if len(dim_list) != 1:
        raise ValueError("Input tensor must have a rank of 1")
      dim_list = noise_estimate.shape.as_list()
      if len(dim_list) != 1:
        raise ValueError("Noise estimate must have a rank of 1")

      snr_shift = 6
      return pcan_fn(input_tensor,
                     noise_estimate,
                     lut_tensor,
                     snr_shift=snr_shift,
                     name=scope)

  return _pcan


# TODO(b/286250473): change back name after name clash resolved
pcan = _pcan_wrapper(gen_pcan_op.signal_pcan, "signal_pcan")

tf.no_gradient("pcan")
