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


def _energy_wrapper(energy_fn, default_name):
  """Wrapper around gen_energy_op.energy*."""

  def _energy(input_tensor, start_index=0, end_index=-1, name=default_name):
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int16)
      dim_list = input_tensor.shape.as_list()
      if len(dim_list) != 1:
        raise ValueError("Input tensor must have a rank of 1")
      if end_index == -1:
        end_index = dim_list[0] - 1
      return energy_fn(input_tensor,
                       start_index=start_index,
                       end_index=end_index,
                       name=name)

  return _energy


# TODO(b/286250473): change back name after name clash resolved
energy = _energy_wrapper(gen_energy_op.signal_energy, "signal_energy")

tf.no_gradient("signal_energy")
