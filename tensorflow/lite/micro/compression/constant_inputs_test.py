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
"""Tests for finding operator inputs that must stay constant."""

import unittest

import numpy as np

from tflite_micro.tensorflow.lite.micro.compression import constant_inputs
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite

PAD = tflite.BuiltinOperator.PAD
PADV2 = tflite.BuiltinOperator.PADV2


class TestFindUses(unittest.TestCase):
  """find_uses() reports the operators that need a tensor constant."""

  def setUp(self):
    self.sg = model_editor.Model().add_subgraph()
    self.const = self.sg.add_tensor(
      shape=(4, 2),
      dtype=tflite.TensorType.INT32,
      data=np.zeros((4, 2), dtype=np.int32),
      name="const",
    )

  def _activation(self) -> model_editor.Tensor:
    return self.sg.add_tensor(shape=(1, 4), dtype=tflite.TensorType.FLOAT32)

  def _add_operator(self, opcode, inputs):
    self.sg.add_operator(
      opcode=opcode, inputs=inputs, outputs=[self._activation()]
    )

  def _add_pads(self, count):
    """Add PAD operators that each take the constant as paddings."""
    for _ in range(count):
      self._add_operator(PAD, [self._activation(), self.const])

  def _find_uses(self):
    return constant_inputs.find_uses(self.sg, self.const)

  def test_finds_a_listed_input(self):
    self._add_pads(1)
    (use,) = self._find_uses()
    self.assertEqual(use.operators, (0,))
    self.assertEqual(use.operator_name, "PAD")
    self.assertEqual(use.position, 1)
    self.assertEqual(use.input_name, "paddings")

  def test_ignores_an_unlisted_input_of_a_listed_operator(self):
    """PAD reads its data input only in Invoke, so input 0 may change."""
    self._add_operator(PAD, [self.const, self._activation()])
    self.assertEqual(self._find_uses(), [])

  def test_ignores_an_unlisted_operator(self):
    self._add_operator(
      tflite.BuiltinOperator.ADD, [self._activation(), self.const]
    )
    self.assertEqual(self._find_uses(), [])

  def test_ignores_operators_reading_other_tensors(self):
    other = self.sg.add_tensor(
      shape=(4, 2),
      dtype=tflite.TensorType.INT32,
      data=np.zeros((4, 2), dtype=np.int32),
    )
    self._add_operator(PAD, [self._activation(), other])
    self.assertEqual(self._find_uses(), [])

  def test_groups_operators_of_one_type(self):
    self._add_pads(3)
    (use,) = self._find_uses()
    self.assertEqual(use.operators, (0, 1, 2))

  def test_separates_operator_types(self):
    self._add_operator(PAD, [self._activation(), self.const])
    self._add_operator(PADV2, [self._activation(), self.const])
    uses = self._find_uses()
    self.assertEqual(
      [(u.operator_name, u.operators) for u in uses],
      [("PAD", (0,)), ("PADV2", (1,))],
    )


if __name__ == "__main__":
  unittest.main()
