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
"""Use overlap add op in python."""

import tensorflow as tf
from tflite_micro.python.tflite_micro.signal.utils import util

gen_overlap_add_op = util.load_custom_op('overlap_add_op.so')


def _overlap_add_wrapper(overlap_add_fn, default_name):
  """Wrapper around gen_overlap_add_op.overlap_add*."""

  def _overlap_add(input_tensor, frame_step, name=default_name):
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor)
      dim_list = input_tensor.shape.as_list()
      if frame_step > dim_list[-1]:
        raise ValueError(
            "Frame_step must not exceed innermost input dimension")
      return overlap_add_fn(input_tensor, frame_step=frame_step, name=name)

  return _overlap_add


# TODO(b/286250473): change back name after name clash resolved
overlap_add = _overlap_add_wrapper(gen_overlap_add_op.signal_overlap_add,
                                   "signal_overlap_add")

tf.no_gradient("signal_overlap_add")
