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


# TODO(b/286250473): change back name after name clash resolved
def stacker(
  input_tensor,
  num_channels,
  stacker_left_context,
  stacker_right_context,
  stacker_step,
  name="signal_stacker",
):
  """Stacks consecutive input frames into a single context window frame.

  On each invocation, accumulates 1-D input frames of size `num_channels` and
  emits a stacked frame containing
  `stacker_left_context + 1 + stacker_right_context` consecutive frames, then
  advances by `stacker_step` frames.

  For example, given a sequence of 10 scalar (`num_channels=1`) frames
  `[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]` with `stacker_left_context=0`,
  `stacker_right_context=1`, and `stacker_step=2`, the stacker produces 5 valid
  stacked frames: `[1, 2]`, `[3, 4]`, `[5, 6]`, `[7, 8]`, `[9, 0]`.

  Args:
    input_tensor: A 1-D `int16` input frame of size `num_channels`.
    num_channels: Positive `int` number of channels in each input frame.
    stacker_left_context: Non-negative `int` number of past frames to include to
      the left of the center frame.
    stacker_right_context: Non-negative `int` number of future frames to include
      to the right of the center frame.
    stacker_step: Positive `int` number of input frames to advance between valid
      outputs.
    name: Optional name for the operation.

  Returns:
    A tuple `(output, output_valid)` where `output` is a 1-D `int16` `Tensor` of
    size `(stacker_left_context + 1 + stacker_right_context) * num_channels` and
    `output_valid` is a scalar `bool` `Tensor` (`True` every `stacker_step`
    frames once enough context frames are buffered, `False` with all-zero
    `output` otherwise).

  Raises:
    ValueError: If `input_tensor` does not have rank 1.
  """
  with tf.name_scope(name) as name:
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int16)
    dim_list = input_tensor.shape.as_list()
    if len(dim_list) != 1:
      raise ValueError("Input tensor must have a rank of 1")

    return gen_stacker_op.signal_stacker(
      input_tensor,
      num_channels=num_channels,
      stacker_left_context=stacker_left_context,
      stacker_right_context=stacker_right_context,
      stacker_step=stacker_step,
      name=name,
    )


tf.no_gradient("signal_stacker")
