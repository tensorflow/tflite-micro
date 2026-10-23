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


# TODO(b/286250473): change back name after name clash resolved
def framer(
  input_tensor, frame_size, frame_step, prefill=False, name="signal_framer"
):
  """Transforms an input signal into overlapping frames of size `frame_size`.

  `frame_step` determines how many input samples the framer advances on each
  invocation. When the framer has accumulated enough samples to emit frames, it
  writes them to `output` and sets `output_valid` to `True`; otherwise `output`
  is uninitialized (zeros) and `output_valid` is `False`.

  For example, given a 10-sample stream `[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]` fed 2
  samples at a time with `frame_size=3`, `frame_step=2`, and `prefill=False`:
    - input `[1, 2]` -> output `[undefined, ...]`, `output_valid=False`
    - input `[3, 4]` -> output `[[1, 2, 3]]`, `output_valid=True`
    - input `[5, 6]` -> output `[[3, 4, 5]]`, `output_valid=True`
    - input `[7, 8]` -> output `[[5, 6, 7]]`, `output_valid=True`
    - input `[9, 0]` -> output `[[7, 8, 9]]`, `output_valid=True`

  Args:
    input_tensor: An `int16` input signal `Tensor` (rank >= 1) whose innermost
      dimension size is a multiple of `frame_step`.
    frame_size: Positive `int` number of samples in each output frame.
    frame_step: Positive `int` number of samples to advance per frame
      (`frame_step <= frame_size`).
    prefill: If `True`, initializes the internal buffer with
      `frame_size - frame_step` zeros so `output_valid` is `True` starting from
      the first invocation. Defaults to `False`.
    name: Optional name for the operation.

  Returns:
    A tuple `(output, output_valid)` where `output` is an `int16` `Tensor` of
    shape `[..., input_samples // frame_step, frame_size]` and `output_valid` is
    a scalar `bool` `Tensor` indicating whether `output` contains valid frames.

  Raises:
    ValueError: If `frame_step > frame_size` or the innermost dimension of
      `input_tensor` is not a multiple of `frame_step`.
  """
  if frame_step > frame_size:
    raise ValueError("frame_step must not be greater than frame_size.")
  with tf.name_scope(name) as name:
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int16)
    dim_list = input_tensor.shape.as_list()
    if dim_list[-1] % frame_step != 0:
      raise ValueError(
        "Innermost input dimension size must be a multiple of %d elements"
        % frame_step
      )
    return gen_framer_op.signal_framer(
      input_tensor,
      frame_size=frame_size,
      frame_step=frame_step,
      prefill=prefill,
      name=name,
    )


tf.no_gradient("signal_framer")
