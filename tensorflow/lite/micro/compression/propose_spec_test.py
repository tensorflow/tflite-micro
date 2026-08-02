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
"""Tests for propose_spec."""

import unittest

import numpy as np

from tflite_micro.tensorflow.lite.micro.compression import propose_spec
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


def build_test_model() -> bytes:
  """Build a model exercising each proposal outcome.

  Tensor 0: activation, no data, never listed.
  Tensor 1: weights with 4 unique values, a shrinkable candidate.
  Tensor 2: bias with 8 unique values in 8 elements, where the value
      table outweighs the shrunken indices, so listed only when savings
      are not required.
  Tensor 3: 200 unique values, unencodable by a 7-bit LUT index.
  Tensor 4: activation, no data, never listed.
  """
  model = model_editor.Model()
  sg = model.add_subgraph()

  act_in = sg.add_tensor(shape=(1, 4),
                         dtype=tflite.TensorType.FLOAT32,
                         name="act_in")
  weights = sg.add_tensor(shape=(8, 4),
                          dtype=tflite.TensorType.FLOAT32,
                          data=np.tile(
                              np.array([-1.0, 0.0, 0.5, 1.0],
                                       dtype=np.float32), 8),
                          name="weights")
  bias = sg.add_tensor(shape=(8, ),
                       dtype=tflite.TensorType.FLOAT32,
                       data=np.arange(8, dtype=np.float32),
                       name="bias")
  lots = sg.add_tensor(shape=(200, ),
                       dtype=tflite.TensorType.FLOAT32,
                       data=np.arange(200, dtype=np.float32),
                       name="lots")
  act_out = sg.add_tensor(shape=(1, 8),
                          dtype=tflite.TensorType.FLOAT32,
                          name="act_out")

  sg.add_operator(opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[act_in, weights, bias],
                  outputs=[act_out])
  sg.inputs = [act_in]
  sg.outputs = [act_out]

  return bytes(model.build())


def entries(text: str) -> dict:
  """Parse a proposal and map (subgraph, tensor) to index_bitwidth."""
  return {
      (t.subgraph, t.tensor): t.compression[0].index_bitwidth
      for t in spec.parse_yaml(text)
  }


class TestProposal(unittest.TestCase):

  def setUp(self):
    self.model = build_test_model()

  def test_lists_shrinkable_constants(self):
    """By default, list only constants that compression shrinks."""
    text = propose_spec.propose(self.model)
    self.assertEqual(entries(text), {(0, 1): 2})

  def test_norequire_savings_lists_all_encodable(self):
    """Without the savings filter, list every LUT-encodable constant."""
    text = propose_spec.propose(self.model, require_savings=False)
    self.assertEqual(entries(text), {(0, 1): 2, (0, 2): 3})

  def test_unencodable_never_listed(self):
    """A tensor with too many unique values is never an entry."""
    text = propose_spec.propose(self.model, require_savings=False)
    self.assertNotIn((0, 3), entries(text))

  def test_footer_explains_rejects(self):
    """Constants left out appear in the footer with a reason."""
    text = propose_spec.propose(self.model)
    footer = [line for line in text.splitlines() if line.startswith("#  ")]
    lots = [line for line in footer if '"lots"' in line]
    self.assertEqual(len(lots), 1)
    self.assertIn("unique values", lots[0])
    bias = [line for line in footer if '"bias"' in line]
    self.assertEqual(len(bias), 1)
    self.assertIn("no savings", bias[0])

  def test_header_estimates_whole_model(self):
    """The header projects the size change of the whole model file."""
    text = propose_spec.propose(self.model)
    self.assertIn(f"# whole model, {len(self.model):,} ->", text)

  def test_comments_identify_tensors(self):
    """Entry comments name the tensor and its consumers."""
    text = propose_spec.propose(self.model)
    self.assertIn('"weights"', text)
    self.assertIn("input 1 of FULLY_CONNECTED (operator 0)", text)

  def test_activations_not_mentioned(self):
    """Tensors without data appear nowhere in the proposal."""
    text = propose_spec.propose(self.model, require_savings=False)
    self.assertNotIn("act_in", text)
    self.assertNotIn("act_out", text)


class TestConsumerGrouping(unittest.TestCase):

  def build(self, consumers: int) -> bytes:
    """Build a model with one constant feeding N ADD operators."""
    model = model_editor.Model()
    sg = model.add_subgraph()
    act = sg.add_tensor(shape=(64, ),
                        dtype=tflite.TensorType.FLOAT32,
                        name="act")
    const = sg.add_tensor(shape=(64, ),
                          dtype=tflite.TensorType.FLOAT32,
                          data=np.tile(np.array([0.0, 1.0], dtype=np.float32),
                                       32),
                          name="const")
    for _ in range(consumers):
      out = sg.add_tensor(shape=(64, ), dtype=tflite.TensorType.FLOAT32)
      sg.add_operator(opcode=tflite.BuiltinOperator.ADD,
                      inputs=[act, const],
                      outputs=[out])
    return bytes(model.build())

  def test_few_consumers_listed_individually(self):
    text = propose_spec.propose(self.build(consumers=2))
    self.assertIn("input 1 of ADD (operators 0, 1)", text)

  def test_many_consumers_summarized(self):
    text = propose_spec.propose(self.build(consumers=5))
    self.assertIn("input 1 of ADD (5 operators)", text)


class TestPerChannel(unittest.TestCase):

  def build(self) -> bytes:
    """Build a model with one per-channel quantized constant.

    Channel 0 holds 2 unique values, channel 1 holds 3, so the bitwidth
    must cover the worst channel with one value table per channel.
    """
    model = model_editor.Model()
    sg = model.add_subgraph()
    row0 = np.tile(np.array([1, 2], dtype=np.int8), 8)
    row1 = np.tile(np.array([3, 4, 5, 3], dtype=np.int8), 4)
    sg.add_tensor(shape=(2, 16),
                  dtype=tflite.TensorType.INT8,
                  data=np.stack([row0, row1]),
                  quantization=model_editor.Quantization(scales=[0.1, 0.2],
                                                         zero_points=[0, 0],
                                                         axis=0),
                  name="weights")
    return bytes(model.build())

  def test_bitwidth_covers_worst_channel(self):
    text = propose_spec.propose(self.build())
    self.assertEqual(entries(text), {(0, 0): 2})
    self.assertIn("2 tables along axis 0", text)


class TestSharedBuffer(unittest.TestCase):

  def build(self) -> bytes:
    """Build a model with two constants sharing one buffer."""
    model = model_editor.Model()
    sg = model.add_subgraph()
    shared = model_editor.Buffer(
        data=np.tile(np.array([1.0, 2.0], dtype=np.float32), 32).tobytes())
    sg.add_tensor(shape=(64, ),
                  dtype=tflite.TensorType.FLOAT32,
                  buffer=shared,
                  name="first")
    sg.add_tensor(shape=(64, ),
                  dtype=tflite.TensorType.FLOAT32,
                  buffer=shared,
                  name="second")
    return bytes(model.build())

  def test_shared_buffer_listed_with_note(self):
    """Tensors sharing a buffer are listed, and each entry names its
    aliases, because the compressor accepts aliases only all together."""
    text = propose_spec.propose(self.build())
    self.assertEqual(entries(text), {(0, 0): 1, (0, 1): 1})
    self.assertIn("shares a buffer with subgraph 0 tensor 1", text)
    self.assertIn("shares a buffer with subgraph 0 tensor 0", text)


class TestEmptyProposal(unittest.TestCase):

  def test_model_without_constants_parses(self):
    """A proposal with no candidates is still a valid, empty spec."""
    model = model_editor.Model()
    sg = model.add_subgraph()
    sg.add_tensor(shape=(1, ), dtype=tflite.TensorType.FLOAT32, name="act")
    text = propose_spec.propose(bytes(model.build()))
    self.assertEqual(spec.parse_yaml(text), [])


if __name__ == "__main__":
  unittest.main()
