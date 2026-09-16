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
  Tensor 1: weights with 4 unique values, a shrinkable candidate. The
      four values recur in every row and every column, so no channel
      axis encodes them smaller than one table does.
  Tensor 2: bias with 8 unique values in 8 elements, where the value
      table outweighs the shrunken indices, so listed only when no
      savings floor applies.
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
                          data=np.stack([
                              np.roll(
                                  np.array([-1.0, 0.0, 0.5, 1.0],
                                           dtype=np.float32), i)
                              for i in range(8)
                          ]),
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


def propose(model: bytes, **kwargs) -> str:
  """Proposes a spec, admitting any entry that shrinks the tensor.

  The fixtures here hold a few dozen bytes each, well under the default
  savings floor, so a test reads better stating the floor it wants than
  inflating its model. TestMinSavings covers the floor itself.
  """
  kwargs.setdefault("min_savings", 1)
  return propose_spec.propose(model, **kwargs)


class TestProposal(unittest.TestCase):

  def setUp(self):
    self.model = build_test_model()

  def test_lists_shrinkable_constants(self):
    """List only constants that compression shrinks."""
    text = propose(self.model)
    self.assertEqual(entries(text), {(0, 1): 2})

  def test_emits_per_tensor_mode(self):
    """A tensor no channel axis encodes smaller proposes per_tensor."""
    text = propose(self.model)
    self.assertIn("per_tensor:", text)
    self.assertNotIn("per_channel:", text)

  def test_zero_floor_lists_all_encodable(self):
    """With no floor, list every LUT-encodable constant."""
    text = propose(self.model, min_savings=0)
    self.assertEqual(entries(text), {(0, 1): 2, (0, 2): 3})

  def test_unencodable_never_listed(self):
    """A tensor with too many unique values is never an entry."""
    text = propose(self.model, min_savings=0)
    self.assertNotIn((0, 3), entries(text))

  def test_footer_explains_rejects(self):
    """Constants left out appear in the footer with a reason."""
    text = propose(self.model)
    footer = [line for line in text.splitlines() if line.startswith("#  ")]
    lots = [line for line in footer if '"lots"' in line]
    self.assertEqual(len(lots), 1)
    self.assertIn("unique values", lots[0])
    bias = [line for line in footer if '"bias"' in line]
    self.assertEqual(len(bias), 1)
    self.assertIn("no savings", bias[0])

  def test_header_estimates_whole_model(self):
    """The header projects the size change of the whole model file."""
    text = propose(self.model)
    self.assertIn(f"# whole model, {len(self.model):,} ->", text)

  def test_comments_identify_tensors(self):
    """Entry comments name the tensor and its consumers."""
    text = propose(self.model)
    self.assertIn('"weights"', text)
    self.assertIn("input 1 of FULLY_CONNECTED (operator 0)", text)

  def test_activations_not_mentioned(self):
    """Tensors without data appear nowhere in the proposal."""
    text = propose(self.model, min_savings=0)
    self.assertNotIn("act_in", text)
    self.assertNotIn("act_out", text)


class TestMinSavings(unittest.TestCase):
  """The floor drops entries too small to earn a DECODE operator."""

  def setUp(self):
    self.model = build_test_model()
    candidates, _ = propose_spec.survey(model_editor.read(self.model),
                                        min_savings=0)
    self.weights = next(c for c in candidates if c.name == '"weights"')

  def test_entry_meeting_the_floor_is_listed(self):
    text = propose(self.model, min_savings=self.weights.savings)
    self.assertIn((0, 1), entries(text))

  def test_entry_under_the_floor_is_dropped(self):
    text = propose(self.model, min_savings=self.weights.savings + 1)
    self.assertNotIn((0, 1), entries(text))

  def test_dropped_entry_reports_its_shortfall(self):
    """The footer separates a shortfall from a tensor that never shrinks."""
    floor = self.weights.savings + 1
    text = propose(self.model, min_savings=floor)
    reasons = [
        line for line in text.splitlines()
        if line.startswith("#  ") and '"weights"' in line
    ]
    self.assertEqual(len(reasons), 1)
    self.assertIn(f"saves {self.weights.savings:,} bytes", reasons[0])
    self.assertIn(f"under the {floor:,} byte floor", reasons[0])
    self.assertNotIn("no savings", reasons[0])

  def test_default_floor_outweighs_a_tiny_entry(self):
    """The default is set high enough to reject a few dozen bytes saved.

    Compression adds an operator and two tensors that the byte estimate
    does not count, so an entry saving less than that structure costs
    grows the model.
    """
    self.assertGreater(propose_spec._DEFAULT_MIN_SAVINGS, self.weights.savings)
    self.assertEqual(entries(propose_spec.propose(self.model)), {})


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
    text = propose(self.build(consumers=2))
    self.assertIn("input 1 of ADD (operators 0, 1)", text)

  def test_many_consumers_summarized(self):
    text = propose(self.build(consumers=5))
    self.assertIn("input 1 of ADD (5 operators)", text)


class TestPerChannel(unittest.TestCase):

  def build(self) -> bytes:
    """Build a model with one constant binned per channel.

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
    text = propose(self.build())
    self.assertEqual(entries(text), {(0, 0): 2})
    self.assertIn("2 tables along axis 0", text)

  def test_emits_per_channel_mode(self):
    """A tensor binned per channel proposes per_channel with its axis."""
    text = propose(self.build())
    self.assertIn("per_channel:", text)
    self.assertIn("axis: 0", text)


class TestModeFromValues(unittest.TestCase):
  """The mode follows the values, not the quantization."""

  def values(self, axis: int) -> np.ndarray:
    """Returns weights binned per channel along the given axis.

    Each of the 70 channels draws from its own pair of values, so the
    whole tensor holds 140 distinct values, more than a 7-bit index can
    enumerate, while any one channel holds two.
    """
    rows = np.stack([
        np.tile(np.array([2 * i, 2 * i + 1], dtype=np.float32), 4)
        for i in range(70)
    ])
    return rows if axis == 0 else rows.T

  def build(self, axis: int = 0, quantization=None) -> bytes:
    model = model_editor.Model()
    sg = model.add_subgraph()
    data = self.values(axis)
    sg.add_tensor(shape=data.shape,
                  dtype=tflite.TensorType.FLOAT32,
                  data=data,
                  quantization=quantization,
                  name="weights")
    return bytes(model.build())

  def test_one_table_cannot_encode_the_tensor(self):
    """Read as one table, the tensor overflows the LUT index."""
    self.assertFalse(propose_spec._encode(self.values(axis=0), None).fits)

  def test_unquantized_proposes_per_channel(self):
    """Binning is found with no quantization to point at it."""
    text = propose(self.build())
    self.assertEqual(entries(text), {(0, 0): 1})
    self.assertIn("per_channel:", text)
    self.assertIn("axis: 0", text)
    self.assertIn("70 tables along axis 0", text)

  def test_single_scale_does_not_force_one_table(self):
    """Per-tensor quantization no longer decides the mode."""
    quantization = model_editor.Quantization(scales=0.5, zero_points=0)
    text = propose(self.build(quantization=quantization))
    self.assertIn("per_channel:", text)
    self.assertIn("axis: 0", text)

  def test_finds_binning_on_the_last_axis(self):
    """The last axis is tested as well as axis 0."""
    text = propose(self.build(axis=1))
    self.assertEqual(entries(text), {(0, 0): 1})
    self.assertIn("axis: 1", text)

  def test_reject_names_the_closest_layout(self):
    """A tensor no layout encodes reports the closest one tried.

    Every row and every column holds 129 distinct values, one more than
    a 7-bit index can enumerate, so neither channel axis rescues it.
    """
    model = model_editor.Model()
    sg = model.add_subgraph()
    grid = np.arange(129).reshape(129, 1) + np.arange(129).reshape(1, 129)
    sg.add_tensor(shape=(129, 129),
                  dtype=tflite.TensorType.INT8,
                  data=(grid % 256 - 128).astype(np.int8),
                  name="dense")
    text = propose(bytes(model.build()))
    footer = [line for line in text.splitlines() if line.startswith("#  ")]
    dense = [line for line in footer if '"dense"' in line]
    self.assertEqual(len(dense), 1)
    self.assertIn("129 unique values per channel along axis 0", dense[0])


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
    text = propose(self.build())
    self.assertEqual(entries(text), {(0, 0): 1, (0, 1): 1})
    self.assertIn("shares a buffer with subgraph 0 tensor 1", text)
    self.assertIn("shares a buffer with subgraph 0 tensor 0", text)


class TestConstantInputs(unittest.TestCase):
  """Tensors a kernel reads in Prepare cannot become DECODE outputs."""

  @classmethod
  def setUpClass(cls):
    """Propose once for a model covering every outcome.

    The paddings tensor holds two unique values in 32 bytes, so it
    shrinks and a savings filter alone would keep it. Every PAD kernel
    requires it constant, which must outrank that. The convolution
    filter and the padded values sit where no kernel requires a
    constant, the first under an operator absent from the list and the
    second at an input position the list does not name.
    """
    super().setUpClass()
    twos = np.array([0.0, 1.0], dtype=np.float32)
    model = model_editor.Model()
    sg = model.add_subgraph()
    act = sg.add_tensor(shape=(1, 6, 6, 4),
                        dtype=tflite.TensorType.FLOAT32,
                        name="act")
    paddings = sg.add_tensor(shape=(4, 2),
                             dtype=tflite.TensorType.INT32,
                             data=np.array([[0, 0], [1, 1], [1, 1], [0, 0]],
                                           dtype=np.int32),
                             name="paddings")
    padded = sg.add_tensor(shape=(1, 8, 8, 4),
                           dtype=tflite.TensorType.FLOAT32,
                           name="padded")
    sg.add_operator(opcode=tflite.BuiltinOperator.PAD,
                    inputs=[act, paddings],
                    outputs=[padded])
    filt = sg.add_tensor(shape=(4, 3, 3, 4),
                         dtype=tflite.TensorType.FLOAT32,
                         data=np.tile(twos, 72).reshape(4, 3, 3, 4),
                         name="filter")
    out = sg.add_tensor(shape=(1, 6, 6, 4),
                        dtype=tflite.TensorType.FLOAT32,
                        name="out")
    sg.add_operator(opcode=tflite.BuiltinOperator.CONV_2D,
                    inputs=[padded, filt],
                    outputs=[out])
    values = sg.add_tensor(shape=(1, 6, 6, 4),
                           dtype=tflite.TensorType.FLOAT32,
                           data=np.tile(twos, 72).reshape(1, 6, 6, 4),
                           name="values")
    padded_values = sg.add_tensor(shape=(1, 8, 8, 4),
                                  dtype=tflite.TensorType.FLOAT32,
                                  name="padded_values")
    sg.add_operator(opcode=tflite.BuiltinOperator.PAD,
                    inputs=[values, paddings],
                    outputs=[padded_values])
    cls.text = propose(bytes(model.build()))
    cls.entries = entries(cls.text)

  def test_required_constant_excluded_by_its_kernel(self):
    """The kernel's requirement outranks the size arithmetic.

    The paddings tensor shrinks, so a proposal that weighed only size
    would keep it.
    """
    self.assertNotIn((0, 1), self.entries)
    reasons = [
        line for line in self.text.splitlines()
        if line.startswith("#  ") and '"paddings"' in line
    ]
    self.assertEqual(len(reasons), 1)
    self.assertIn("must stay constant", reasons[0])
    self.assertIn("PAD", reasons[0])
    self.assertNotIn("no savings", reasons[0])

  def test_unlisted_operator_still_proposed(self):
    """A convolution filter stays compressible.

    Only kernels built for particular targets read a filter in
    Prepare, so the list leaves convolution weights alone.
    """
    self.assertIn((0, 3), self.entries)

  def test_unlisted_input_position_still_proposed(self):
    """The requirement binds one input position, not the operator.

    PAD requires its paddings, at input 1. A constant reaching PAD at
    another position is compressible.
    """
    self.assertIn((0, 5), self.entries)


class TestEmptyProposal(unittest.TestCase):

  def test_model_without_constants_parses(self):
    """A proposal with no candidates is still a valid, empty spec."""
    model = model_editor.Model()
    sg = model.add_subgraph()
    sg.add_tensor(shape=(1, ), dtype=tflite.TensorType.FLOAT32, name="act")
    text = propose(bytes(model.build()))
    self.assertEqual(spec.parse_yaml(text), [])


if __name__ == "__main__":
  unittest.main()
