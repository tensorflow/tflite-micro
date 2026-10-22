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
"""Use energy op in python."""

import tensorflow as tf
from tflite_micro.python.tflite_micro.signal.utils import util

gen_energy_op = util.load_custom_op('energy_op.so')


# TODO(b/286250473): change back name after name clash resolved
def energy(input_tensor, start_index=0, end_index=-1, name="signal_energy"):
  """Calculates the spectral energy of a complex frequency-domain array.

  Only elements in the complex bin index range `[start_index, end_index]` are
  calculated, and the remaining elements are set to zero.

  Args:
    input_tensor: A 1-D `int16` frequency-domain signal containing interleaved
      real and imaginary components of complex spectral bins.
    start_index: Complex bin index in `input_tensor` to start calculating from.
      Defaults to `0`.
    end_index: Last complex bin index to calculate, or `-1` to use the end of
      `input_tensor`. Defaults to `-1`.
    name: Optional name for the operation.

  Returns:
    A 1-D `uint32` `Tensor` of length `len(input_tensor) // 2` containing the
    squared magnitude of each complex bin in `[start_index, end_index]`.

  Raises:
    ValueError: If `input_tensor` does not have rank 1.
  """
  with tf.name_scope(name) as name:
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int16)
    dim_list = input_tensor.shape.as_list()
    if len(dim_list) != 1:
      raise ValueError("Input tensor must have a rank of 1")
    if end_index == -1:
      end_index = dim_list[0] - 1
    return gen_energy_op.signal_energy(
      input_tensor, start_index=start_index, end_index=end_index, name=name
    )


tf.no_gradient("signal_energy")
