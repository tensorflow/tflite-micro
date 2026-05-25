# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for verify.py.

The library's callers only ever exercise the passing direction, so
these tests focus on the failing direction: assert_outputs_match must
raise when two models disagree. The fixture model has two inputs and
two outputs, each input feeding its own operator, so the input and
output loops genuinely iterate and a positional mix-up between the
branches shows up as a mismatch.
"""

import unittest

import numpy as np

from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import verify
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


class VerifyTest(unittest.TestCase):

  def test_identical_models_pass(self):
    verify.assert_outputs_match(_build_model(), _build_model())

  def test_mismatch_in_either_branch_raises(self):
    for branch in (1, 2):
      with self.subTest(branch=branch):
        with self.assertRaises(AssertionError):
          verify.assert_outputs_match(_build_model(),
                                      _build_model(nudge_branch=branch))

  def test_tolerance_allows_small_differences(self):
    original = _build_model()
    nudged = _build_model(nudge_branch=2)

    with self.assertRaises(AssertionError):
      verify.assert_outputs_match(original, nudged)

    # int8 outputs differ by at most 255, so this tolerance admits the
    # nudge while still proving the tolerance parameter changes the
    # outcome of the exact comparison above.
    verify.assert_outputs_match(original,
                                nudged,
                                tolerance=verify.Tolerance(rtol=0, atol=255))


def _build_model(nudge_branch=None):
  """Build a model holding two disconnected one-layer branches.

  The graph has two dataflow paths that share nothing:

      input1 -> FULLY_CONNECTED(weights1) -> output1
      input2 -> FULLY_CONNECTED(weights2) -> output2

  One invocation runs both branches. This gives the model two inputs
  and two outputs, with each output depending on exactly one input
  and one weight tensor, so a comparison that mixes up input or
  output positions compares genuinely different numbers. The two
  weight tensors differ from each other for the same reason.

  nudge_branch=1 or 2 adds one to a single weight in that branch,
  making only that branch's output differ from the unnudged model's.
  """
  # 4 unique small values per tensor avoid saturation; different rows
  # produce varied outputs.
  weights_data = [
      np.array([
          [-1, 0, 0, 1],
          [-1, 0, 1, 1],
          [-1, 1, 1, 1],
          [0, 1, 1, 1],
      ],
               dtype=np.int8),
      np.array([
          [1, 1, 1, 1],
          [1, 1, 2, 2],
          [1, 2, 2, 3],
          [2, 2, 3, 3],
      ],
               dtype=np.int8),
  ]
  if nudge_branch is not None:
    nudged = weights_data[nudge_branch - 1].copy()
    nudged[0, 0] += 1
    weights_data[nudge_branch - 1] = nudged

  def tensor(name, shape, data=None):
    return model_editor.Tensor(shape=shape,
                               dtype=tflite.TensorType.INT8,
                               data=data,
                               name=name,
                               quantization=model_editor.Quantization(
                                   scales=1.0, zero_points=0))

  return model_editor.Model(subgraphs=[
      model_editor.Subgraph(
          tensors=[
              w1 := tensor("weights1", (4, 4), weights_data[0]),
              w2 := tensor("weights2", (4, 4), weights_data[1]),
          ],
          inputs=[
              i1 := tensor("input1", (1, 4)),
              i2 := tensor("input2", (1, 4)),
          ],
          outputs=[
              o1 := tensor("output1", (1, 4)),
              o2 := tensor("output2", (1, 4)),
          ],
          operators=[
              model_editor.Operator(
                  opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[i1, w1],
                  outputs=[o1]),
              model_editor.Operator(
                  opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[i2, w2],
                  outputs=[o2]),
          ],
      )
  ]).build()


if __name__ == "__main__":
  unittest.main()
