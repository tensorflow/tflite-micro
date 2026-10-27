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


# TODO(b/286250473): change back name after name clash resolved
def overlap_add(input_tensor, frame_step, name="signal_overlap_add"):
  """Reconstructs a signal from overlapping frames spaced `frame_step` apart.

  Transforms an input made of overlapping frames of length `frame_size` into an
  output signal by shifting successive frames by `frame_step` samples and
  summing aligned samples.

  For example, for 5 input frames with `frame_size=3` and `frame_step=2`
  (`[1, 2, 3]`, `[4, 5, 6]`, `[7, 8, 9]`, `[0, 1, 2]`, `[3, 4, 5]`):
  ```
  [1, 2, 3]
        [4, 5, 6]
              [7, 8, 9]
                    [0, 1, 2]
                          [3, 4, 5]
  ```
  Adding vertically aligned samples produces 5 output steps of `frame_step=2`
  samples each: `[1, 2]`, `[7, 5]`, `[13, 8]`, `[9, 1]`, `[5, 4]`.

  Args:
    input_tensor: A `float32` or `int16` `Tensor` of shape
      `[..., frames, frame_size]` with rank >= 2.
    frame_step: Positive `int` step size in samples (`frame_step <= frame_size`)
      and number of output samples produced per input frame.
    name: Optional name for the operation.

  Returns:
    A `Tensor` of the same dtype as `input_tensor` with shape
    `[..., frames * frame_step]` containing the overlap-added output samples.

  Raises:
    ValueError: If `frame_step` exceeds the innermost dimension of
      `input_tensor`.
  """
  with tf.name_scope(name) as name:
    input_tensor = tf.convert_to_tensor(input_tensor)
    dim_list = input_tensor.shape.as_list()
    if frame_step > dim_list[-1]:
      raise ValueError("Frame_step must not exceed innermost input dimension")
    return gen_overlap_add_op.signal_overlap_add(
      input_tensor, frame_step=frame_step, name=name
    )


tf.no_gradient("signal_overlap_add")
