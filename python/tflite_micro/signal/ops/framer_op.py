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
"""Use framer op in python."""

import tensorflow as tf
from tflite_micro.python.tflite_micro.signal.utils import util

gen_framer_op = util.load_custom_op('framer_op.so')


def _framer_wrapper(framer_fn, default_name):
  """Wrapper around gen_framer_op.framer*."""

  def _framer(input_tensor,
              frame_size,
              frame_step,
              prefill=False,
              name=default_name):
    if frame_step > frame_size:
      raise ValueError("frame_step must not be greater than frame_size.")
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int16)
      dim_list = input_tensor.shape.as_list()
      if dim_list[-1] % frame_step != 0:
        raise ValueError(
            "Innermost input dimenion size must be a multiple of %d elements" %
            frame_step)
      return framer_fn(input_tensor,
                       frame_size=frame_size,
                       frame_step=frame_step,
                       prefill=prefill,
                       name=name)

  return _framer


# TODO(b/286250473): change back name after name clash resolved
framer = _framer_wrapper(gen_framer_op.signal_framer, "signal_framer")

tf.no_gradient("signal_framer")
