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
"""Use stacker op in python."""

import tensorflow as tf
from tflite_micro.python.tflite_micro.signal.utils import util

gen_stacker_op = util.load_custom_op('stacker_op.so')


def _stacker_wrapper(stacker_fn, default_name):
  """Wrapper around gen_stacker_op.stacker*."""

  def _stacker(input_tensor,
               num_channels,
               stacker_left_context,
               stacker_right_context,
               stacker_step,
               name=default_name):
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int16)
      dim_list = input_tensor.shape.as_list()
      if len(dim_list) != 1:
        raise ValueError("Input tensor must have a rank of 1")

      return stacker_fn(input_tensor,
                        num_channels=num_channels,
                        stacker_left_context=stacker_left_context,
                        stacker_right_context=stacker_right_context,
                        stacker_step=stacker_step,
                        name=name)

  return _stacker


# TODO(b/286250473): change back name after name clash resolved
stacker = _stacker_wrapper(gen_stacker_op.signal_stacker, "signal_stacker")

tf.no_gradient("signal_stacker")
