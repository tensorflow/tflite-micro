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

import contextlib
import io
import os
import tempfile
import unittest
import warnings

import numpy as np

from tflite_micro.tensorflow.lite.micro.compression import compress
from tflite_micro.tensorflow.lite.micro.compression import lut
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import propose_spec
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite

FLOAT32 = tflite.TensorType.FLOAT32


def pair_tiled(shape=(64,), low=0.0, high=1.0) -> np.ndarray:
  """Returns floats alternating between two values, which LUT shrinks."""
  return np.resize(np.array([low, high], dtype=np.float32), shape)


def add_reader(sg: model_editor.Subgraph, tensor: model_editor.Tensor):
  """Adds an operator reading the tensor, so compression can decode it."""
  out = sg.add_tensor(shape=tensor.shape, dtype=tensor.dtype)
  sg.add_operator(
    opcode=tflite.BuiltinOperator.ABS, inputs=[tensor], outputs=[out]
  )


# PAD's paddings for a 4-D input: one row and one column on each side.
PADDINGS = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)


def add_pad(
  sg: model_editor.Subgraph,
  tensor: model_editor.Tensor,
  paddings: model_editor.Tensor,
) -> model_editor.Tensor:
  """Adds a PAD operator reading the tensor and paddings, and returns its
  output."""
  shape = tuple(d + sum(p) for d, p in zip(tensor.shape, PADDINGS))
  padded = sg.add_tensor(shape=shape, dtype=tensor.dtype)
  sg.add_operator(
    opcode=tflite.BuiltinOperator.PAD,
    inputs=[tensor, paddings],
    outputs=[padded],
  )
  return padded


def build_single_constant(
  data, dtype=FLOAT32, shape=None, quantization=None, name="weights"
) -> bytes:
  """Build a model with one constant, tensor 0, read by one operator."""
  model = model_editor.Model()
  sg = model.add_subgraph()
  tensor = sg.add_tensor(
    shape=data.shape if shape is None else shape,
    dtype=dtype,
    data=data,
    quantization=quantization,
    name=name,
  )
  add_reader(sg, tensor)
  return bytes(model.build())


# Four values that recur in every row and every column of a 32x16 tensor,
# so no channel axis encodes them smaller than one table does.
ROLLED = np.stack(
  [
    np.roll(np.tile(np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32), 4), i)
    for i in range(32)
  ]
)


def build_test_model() -> bytes:
  """Build a model with one listed constant, one that compression grows,
  and one that no layout encodes.

  Tensor 0: activation, no data, never listed.
  Tensor 1: weights holding ROLLED, a shrinkable candidate.
  Tensor 2: bias with 32 unique values in 32 elements, where the value
      table outweighs the shrunken indices, so listed only when no
      savings floor applies.
  Tensor 3: 200 unique values, unencodable by a 7-bit LUT index, read
      by an ABS operator.
  Tensor 4: activation, no data, never listed.
  Tensor 5: output of the ABS operator, no data, never listed.

  Tensors 2 and 3 are one-dimensional, so each channel slice would hold
  one element, and only the per-tensor layout is tested.
  """
  model = model_editor.Model()
  sg = model.add_subgraph()

  act_in = sg.add_tensor(shape=(1, 16), dtype=FLOAT32, name="act_in")
  weights = sg.add_tensor(
    shape=(32, 16), dtype=FLOAT32, data=ROLLED, name="weights"
  )
  bias = sg.add_tensor(
    shape=(32,),
    dtype=FLOAT32,
    data=np.arange(32, dtype=np.float32),
    name="bias",
  )
  lots = sg.add_tensor(
    shape=(200,),
    dtype=FLOAT32,
    data=np.arange(200, dtype=np.float32),
    name="lots",
  )
  act_out = sg.add_tensor(shape=(1, 32), dtype=FLOAT32, name="act_out")

  sg.add_operator(
    opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
    inputs=[act_in, weights, bias],
    outputs=[act_out],
  )
  add_reader(sg, lots)
  sg.inputs = [act_in]
  sg.outputs = [act_out]

  return bytes(model.build())


def per_channel_values() -> np.ndarray:
  """Returns int8 weights whose channel 0 holds 2 unique values and
  channel 1 holds 3, so each channel needs its own value table."""
  row0 = np.tile(np.array([1, 2], dtype=np.int8), 256)
  row1 = np.tile(np.array([3, 4, 5, 3], dtype=np.int8), 128)
  return np.stack([row0, row1])


def binned_values(channels: int = 70, axis: int = 0) -> np.ndarray:
  """Returns weights binned per channel along the given axis.

  Each channel draws from its own pair of values, so the whole tensor
  holds twice as many unique values as it has channels, while any one
  channel holds two. With 70 channels, the whole tensor holds 140,
  more than a 7-bit index can address.
  """
  rows = np.stack(
    [pair_tiled((64,), 2 * i, 2 * i + 1) for i in range(channels)]
  )
  return rows if axis == 0 else rows.T


def build_shared_buffer_model() -> bytes:
  """Build a model with two constants, tensors 0 and 1, sharing a buffer."""
  model = model_editor.Model()
  sg = model.add_subgraph()
  shared = model_editor.Buffer(data=pair_tiled(low=1.0, high=2.0).tobytes())
  first = sg.add_tensor(shape=(64,), dtype=FLOAT32, buffer=shared, name="first")
  second = sg.add_tensor(
    shape=(64,), dtype=FLOAT32, buffer=shared, name="second"
  )
  add_reader(sg, first)
  add_reader(sg, second)
  return bytes(model.build())


def build_unlisted_alias_model() -> bytes:
  """Build a model where an unread constant, tensor 0, shares a buffer
  with a constant, tensor 1, that would be listed on its own merits."""
  model = model_editor.Model()
  sg = model.add_subgraph()
  shared = model_editor.Buffer(data=pair_tiled().tobytes())
  sg.add_tensor(shape=(64,), dtype=FLOAT32, buffer=shared, name="unread")
  alias = sg.add_tensor(shape=(64,), dtype=FLOAT32, buffer=shared, name="alias")
  add_reader(sg, alias)
  return bytes(model.build())


def build_two_subgraph_model() -> bytes:
  """Build a model with a constant in each of two subgraphs, plus a
  buffer that one tensor in each subgraph shares.

  Subgraph 0: tensor 0 "first", tensor 2 "shared0".
  Subgraph 1: tensor 0 "second", tensor 2 "shared1".
  """
  model = model_editor.Model()
  shared = model_editor.Buffer(data=pair_tiled(low=4.0, high=5.0).tobytes())
  for i, name in enumerate(("first", "second")):
    sg = model.add_subgraph()
    own = sg.add_tensor(
      shape=(64,),
      dtype=FLOAT32,
      data=pair_tiled(low=2.0 * i, high=2.0 * i + 1),
      name=name,
    )
    add_reader(sg, own)
    alias = sg.add_tensor(
      shape=(64,), dtype=FLOAT32, buffer=shared, name=f"shared{i}"
    )
    add_reader(sg, alias)
  return bytes(model.build())


def build_quantized_readers_model() -> bytes:
  """Build a model with per-channel quantized weights, tensor 0, read by
  three operators, each of which gets a decoded copy of the quantization."""
  model = model_editor.Model()
  sg = model.add_subgraph()
  weights = sg.add_tensor(
    shape=(64, 64),
    dtype=tflite.TensorType.INT8,
    data=np.resize(np.array([1, 2], dtype=np.int8), (64, 64)),
    quantization=model_editor.Quantization(
      scales=[0.5] * 64, zero_points=[0] * 64, axis=0
    ),
    name="weights",
  )
  for _ in range(3):
    add_reader(sg, weights)
  return bytes(model.build())


def build_two_inputs_model() -> bytes:
  """Build a model with one operator reading two constants, tensors 0 and
  1, which one DECODE decodes together."""
  model = model_editor.Model()
  sg = model.add_subgraph()
  a = sg.add_tensor(shape=(64,), dtype=FLOAT32, data=pair_tiled(), name="a")
  b = sg.add_tensor(
    shape=(64,), dtype=FLOAT32, data=pair_tiled(low=2.0, high=3.0), name="b"
  )
  out = sg.add_tensor(shape=(64,), dtype=FLOAT32)
  sg.add_operator(
    opcode=tflite.BuiltinOperator.ADD, inputs=[a, b], outputs=[out]
  )
  return bytes(model.build())


def entries(text: str) -> dict:
  """Parse a proposal into {(subgraph, tensor): (index_bitwidth, axis)}.

  The axis is None for a per-tensor entry.
  """
  result = {}
  for t in spec.parse_yaml(text):
    method = t.compression[0]
    axis = (
      method.mode.axis if isinstance(method.mode, spec.PerChannel) else None
    )
    result[(t.subgraph, t.tensor)] = (method.index_bitwidth, axis)
  return result


def footer_line(text: str, name: str) -> str:
  """Returns the one footer line naming the tensor."""
  lines = [
    line
    for line in text.splitlines()
    if line.startswith("#  ") and f'"{name}"' in line
  ]
  if len(lines) != 1:
    raise AssertionError(f"expected one footer line for {name}: {lines}")
  return lines[0]


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
    """List only constants that compression shrinks, per tensor when no
    channel axis encodes them smaller."""
    self.assertEqual(entries(propose(self.model)), {(0, 1): (2, None)})

  def test_zero_floor_lists_all_encodable(self):
    """With no floor, list every LUT-encodable constant."""
    text = propose(self.model, min_savings=0)
    self.assertEqual(entries(text), {(0, 1): (2, None), (0, 2): (5, None)})

  def test_footer_explains_rejects(self):
    """Constants left out appear in the footer with a reason."""
    text = propose(self.model)
    self.assertIn("unique values", footer_line(text, "lots"))
    self.assertIn("no savings", footer_line(text, "bias"))

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
  """The floor drops entries whose savings, after the DECODE operators
  and tensors they add, fall under it."""

  def setUp(self):
    self.model = build_test_model()
    candidates, _ = propose_spec.survey(
      model_editor.read(self.model), min_savings=0
    )
    self.weights = next(c for c in candidates if c.name == '"weights"')

  def test_entry_meeting_the_floor_is_listed(self):
    text = propose(self.model, min_savings=self.weights.net_savings)
    self.assertIn((0, 1), entries(text))

  def test_entry_under_the_floor_is_dropped(self):
    text = propose(self.model, min_savings=self.weights.net_savings + 1)
    self.assertNotIn((0, 1), entries(text))

  def test_dropped_entry_reports_its_shortfall(self):
    """The footer separates a shortfall from a tensor that never shrinks."""
    floor = self.weights.net_savings + 1
    reason = footer_line(propose(self.model, min_savings=floor), "weights")
    self.assertIn(f"saves {self.weights.net_savings:,} bytes", reason)
    self.assertIn(f"under the {floor:,} byte floor", reason)
    self.assertNotIn("no savings", reason)

  def test_default_floor_rejects_a_small_net_saving(self):
    """By default, an entry must save more than a small net gain."""
    model = build_single_constant(pair_tiled())
    (candidate,), _ = propose_spec.survey(
      model_editor.read(model), min_savings=0
    )
    self.assertGreater(candidate.net_savings, 0)
    self.assertEqual(entries(propose_spec.propose(model)), {})


class TestConsumerGrouping(unittest.TestCase):
  def test_operators_reading_one_input_share_a_description(self):
    """Operators of one type reading the tensor at one input position
    appear in one description, not one each."""
    model = model_editor.Model()
    sg = model.add_subgraph()
    act = sg.add_tensor(shape=(256,), dtype=FLOAT32, name="act")
    const = sg.add_tensor(
      shape=(256,), dtype=FLOAT32, data=pair_tiled((256,)), name="const"
    )
    for _ in range(2):
      out = sg.add_tensor(shape=(256,), dtype=FLOAT32)
      sg.add_operator(
        opcode=tflite.BuiltinOperator.ADD, inputs=[act, const], outputs=[out]
      )
    text = propose(bytes(model.build()))
    self.assertEqual(text.count("input 1 of ADD"), 1)


class TestPerChannel(unittest.TestCase):
  def test_bitwidth_covers_the_largest_table(self):
    """The per-channel bitwidth addresses the channel with the most
    unique values."""
    text = propose(
      build_single_constant(per_channel_values(), tflite.TensorType.INT8)
    )
    self.assertEqual(entries(text), {(0, 0): (2, 0)})
    self.assertIn("2 tables along axis 0", text)


class TestModeFromValues(unittest.TestCase):
  """The mode follows the values, not the quantization."""

  def test_one_table_cannot_encode_the_tensor(self):
    """Read as one table, the fixture overflows the LUT index."""
    unique = len(np.unique(binned_values()))
    self.assertGreater(unique, 2**lut.LutAncillaryData.MAX_BITWIDTH)

  def test_unquantized_proposes_per_channel(self):
    """Binning is found with no quantization to point at it."""
    text = propose(build_single_constant(binned_values()))
    self.assertEqual(entries(text), {(0, 0): (1, 0)})
    self.assertIn("70 tables along axis 0", text)

  def test_finds_binning_on_the_last_axis(self):
    """The last axis is tested as well as axis 0."""
    text = propose(build_single_constant(binned_values(axis=1)))
    self.assertEqual(entries(text), {(0, 0): (1, 1)})

  def test_single_scale_does_not_force_one_table(self):
    """Per-tensor quantization does not choose one table when a channel
    axis encodes the values smaller.

    The 8 channels hold 16 unique values in all, which one table can
    address, but one table per channel needs only 1-bit indices.
    """
    quantization = model_editor.Quantization(scales=0.5, zero_points=0)
    model = build_single_constant(
      binned_values(channels=8), quantization=quantization
    )
    self.assertEqual(entries(propose(model)), {(0, 0): (1, 0)})

  def test_channel_scales_do_not_force_per_channel(self):
    """Per-channel quantization does not choose per-channel tables when
    one table encodes the values smaller."""
    quantization = model_editor.Quantization(
      scales=[0.5] * 32, zero_points=[0] * 32, axis=0
    )
    model = build_single_constant(ROLLED, quantization=quantization)
    self.assertEqual(entries(propose(model)), {(0, 0): (2, None)})

  def test_tie_goes_to_one_table(self):
    """When a channel axis encodes no smaller, propose one table."""
    model = build_single_constant(pair_tiled((1, 64)))
    self.assertEqual(entries(propose(model)), {(0, 0): (1, None)})


class TestRejectReasons(unittest.TestCase):
  def test_reject_names_the_closest_layout(self):
    """A tensor no layout encodes reports the closest one tried.

    Every row and every column holds 129 unique values, one more than
    a 7-bit index can address, so neither channel axis rescues it.
    """
    grid = np.arange(129).reshape(129, 1) + np.arange(129).reshape(1, 129)
    model = build_single_constant(
      (grid % 256 - 128).astype(np.int8), tflite.TensorType.INT8, name="dense"
    )
    self.assertIn(
      "129 unique values per channel along axis 0",
      footer_line(propose(model), "dense"),
    )

  def test_unreadable_type_left_out(self):
    """A type with no numpy equivalent cannot be read, so it cannot be
    compressed."""
    model = build_single_constant(
      bytes(16), tflite.TensorType.INT4, shape=(32,), name="packed"
    )
    self.assertIn("no numpy dtype", footer_line(propose(model), "packed"))


class TestSharedBuffer(unittest.TestCase):
  def test_shared_buffer_listed_with_note(self):
    """List each alias, and name the others in its comment.

    The compressor accepts the aliases of a buffer only all together.
    """
    text = propose(build_shared_buffer_model())
    self.assertEqual(entries(text), {(0, 0): (1, None), (0, 1): (1, None)})
    self.assertIn("shares a buffer with subgraph 0 tensor 1", text)
    self.assertIn("shares a buffer with subgraph 0 tensor 0", text)

  def test_totals_count_the_buffer_once(self):
    """Totals count a shared buffer and its identical encodings once.

    The model stores a shared buffer once, and aliases of one shape
    compress to identical data, also stored once.
    """
    model = build_shared_buffer_model()
    candidates, _ = propose_spec.survey(model_editor.read(model), min_savings=1)
    estimated = candidates[0].estimated_bytes
    text = propose(model)
    self.assertIn(f"# 2 tensors, 256 -> {estimated:,} bytes", text)

  def test_alias_of_an_unlisted_tensor_is_left_out(self):
    """Compressing some aliases of a buffer leaves its data in the model."""
    text = propose(build_unlisted_alias_model())
    self.assertEqual(entries(text), {})
    self.assertIn(
      "shares a buffer with subgraph 0 tensor 0, which is not listed",
      footer_line(text, "alias"),
    )


class TestSubgraphs(unittest.TestCase):
  def test_entries_name_their_subgraph(self):
    text = propose(build_two_subgraph_model())
    self.assertEqual(set(entries(text)), {(0, 0), (0, 2), (1, 0), (1, 2)})

  def test_alias_across_subgraphs_noted(self):
    text = propose(build_two_subgraph_model())
    self.assertIn("shares a buffer with subgraph 1 tensor 2", text)
    self.assertIn("shares a buffer with subgraph 0 tensor 2", text)


class TestConstantInputs(unittest.TestCase):
  """Tensors a kernel reads in Prepare cannot become DECODE outputs."""

  @classmethod
  def setUpClass(cls):
    """Propose once for a model with a required constant and two that
    are not.

    With no floor, size alone would list the paddings tensor. Every PAD
    kernel requires it constant, so the proposal leaves it out. The
    convolution filter and the
    padded values are inputs where no kernel requires a constant.
    CONV_2D is not in the list, and PAD's input 0 is not a listed
    position.
    """
    super().setUpClass()
    model = model_editor.Model()
    sg = model.add_subgraph()
    act = sg.add_tensor(shape=(1, 6, 6, 4), dtype=FLOAT32, name="act")
    paddings = sg.add_tensor(
      shape=(4, 2),
      dtype=tflite.TensorType.INT32,
      data=PADDINGS,
      name="paddings",
    )
    padded = add_pad(sg, act, paddings)
    filt = sg.add_tensor(
      shape=(4, 3, 3, 4),
      dtype=FLOAT32,
      data=pair_tiled((4, 3, 3, 4)),
      name="filter",
    )
    out = sg.add_tensor(shape=(1, 6, 6, 4), dtype=FLOAT32, name="out")
    sg.add_operator(
      opcode=tflite.BuiltinOperator.CONV_2D,
      inputs=[padded, filt],
      outputs=[out],
    )
    values = sg.add_tensor(
      shape=(1, 6, 6, 4),
      dtype=FLOAT32,
      data=pair_tiled((1, 6, 6, 4)),
      name="values",
    )
    add_pad(sg, values, paddings)
    cls.text = propose(bytes(model.build()), min_savings=0)
    cls.entries = entries(cls.text)

  def test_required_constant_excluded_by_its_kernel(self):
    """The kernel's requirement takes precedence over the size arithmetic.

    With no floor, a proposal that weighed only size would keep the
    paddings tensor.
    """
    self.assertNotIn((0, 1), self.entries)
    reason = footer_line(self.text, "paddings")
    self.assertIn("must stay constant", reason)
    self.assertIn("PAD", reason)
    self.assertNotIn("no savings", reason)

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


class TestReaders(unittest.TestCase):
  """Compression decodes a tensor only where something reads it."""

  def build(self, as_output: bool) -> bytes:
    """Build a model with one constant that no operator reads."""
    model = model_editor.Model()
    sg = model.add_subgraph()
    const = sg.add_tensor(
      shape=(64,), dtype=FLOAT32, data=pair_tiled(), name="const"
    )
    if as_output:
      sg.outputs = [const]
    return bytes(model.build())

  def test_unread_constant_left_out(self):
    text = propose(self.build(as_output=False))
    self.assertEqual(entries(text), {})
    self.assertIn("no operator reads it", footer_line(text, "const"))

  def test_subgraph_output_listed(self):
    text = propose(self.build(as_output=True))
    self.assertEqual(entries(text), {(0, 0): (1, None)})
    self.assertIn("subgraph output", text)


class TestVariableTensor(unittest.TestCase):
  def test_variable_never_mentioned(self):
    """A variable holds kernel state, so it is not a constant."""
    model = model_editor.Model()
    sg = model.add_subgraph()
    state = sg.add_tensor(
      shape=(64,), dtype=FLOAT32, data=pair_tiled(), name="state"
    )
    state.is_variable = True
    add_reader(sg, state)
    text = propose(bytes(model.build()), min_savings=0)
    self.assertEqual(entries(text), {})
    self.assertNotIn("state", text)


class TestTensorNames(unittest.TestCase):
  def test_newline_in_a_name_keeps_the_spec_valid(self):
    """A name is escaped, so it cannot end its comment line early."""
    model = model_editor.Model()
    sg = model.add_subgraph()
    listed = sg.add_tensor(
      shape=(256,),
      dtype=FLOAT32,
      data=pair_tiled((256,)),
      name="listed\ntensors: []",
    )
    add_reader(sg, listed)
    sg.add_tensor(
      shape=(256,),
      dtype=FLOAT32,
      data=np.arange(256, dtype=np.float32),
      name="unread\ntensors: []",
    )
    text = propose(bytes(model.build()))
    self.assertEqual(entries(text), {(0, 0): (1, None)})


class TestEmptyProposal(unittest.TestCase):
  def test_model_without_constants_parses(self):
    """A proposal with no candidates is still a valid, empty spec."""
    model = model_editor.Model()
    sg = model.add_subgraph()
    sg.add_tensor(shape=(1,), dtype=FLOAT32, name="act")
    text = propose(bytes(model.build()))
    self.assertEqual(spec.parse_yaml(text), [])


class TestCompressorAgreement(unittest.TestCase):
  """The proposal matches what the compressor does with it."""

  MODELS = {
    "test": build_test_model,
    "per_channel": lambda: build_single_constant(
      per_channel_values(), tflite.TensorType.INT8
    ),
    "binned_axis_0": lambda: build_single_constant(binned_values()),
    "binned_axis_1": lambda: build_single_constant(binned_values(axis=1)),
    "shared_buffer": build_shared_buffer_model,
    "two_subgraphs": build_two_subgraph_model,
    "quantized_readers": build_quantized_readers_model,
    "two_inputs": build_two_inputs_model,
  }

  # Padding and layout in a model file vary with its other contents, so
  # a tensor compressed in its own model and in the full model can save
  # a few bytes more in one than the other (up to 20 in these models).
  LAYOUT_SLOP = 32

  def test_estimates_match_the_compressor(self):
    """Each estimated size equals the encoded and ancillary data the LUT
    compressor produces."""
    for label, build in self.MODELS.items():
      with self.subTest(model=label):
        model_bytes = build()
        model = model_editor.read(model_bytes)
        candidates, _ = propose_spec.survey(model, min_savings=0)
        methods = {
          (t.subgraph, t.tensor): t.compression[0]
          for t in spec.parse_yaml(propose(model_bytes, min_savings=0))
        }
        for c in candidates:
          tensor = model.subgraphs[c.subgraph].tensors[c.tensor]
          result = lut.LutCompressor().compress(
            tensor, methods[(c.subgraph, c.tensor)]
          )
          self.assertEqual(
            len(result.encoded_data) + len(result.ancillary_data),
            c.estimated_bytes,
          )

  def test_compressor_accepts_every_entry(self):
    """The compressor takes the proposal without dropping an entry or
    warning about one."""
    for label, build in self.MODELS.items():
      with self.subTest(model=label):
        model = build()
        specs = spec.parse_yaml(propose(model))
        with warnings.catch_warnings(record=True) as caught:
          warnings.simplefilter("always")
          compress.compress(model, specs)
        self.assertEqual([str(w.message) for w in caught], [])

  def test_whole_model_line_is_the_compressed_size(self):
    """The header gives the size of the model compressed as proposed."""
    for label, build in self.MODELS.items():
      with self.subTest(model=label):
        model = build()
        text = propose(model)
        actual = len(compress.compress(model, spec.parse_yaml(text)))
        self.assertIn(f"-> {actual:,} bytes", text)

  def test_net_savings_match_compressing_the_entry_alone(self):
    """Each entry's net savings equal what compressing it alone in the
    model saves, DECODE operators and tensors included."""
    for label, build in self.MODELS.items():
      with self.subTest(model=label):
        model_bytes = build()
        candidates, _ = propose_spec.survey(
          model_editor.read(model_bytes), min_savings=0
        )
        for c in candidates:
          if c.sharers:
            # The compressor takes the aliases of a buffer only together.
            continue
          entry = c.spec_entry()
          with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compressed = compress.compress(model_bytes, [entry])
          self.assertAlmostEqual(
            c.net_savings,
            len(model_bytes) - len(compressed),
            delta=self.LAYOUT_SLOP,
          )


class TestMain(unittest.TestCase):
  def setUp(self):
    self.dir = tempfile.TemporaryDirectory()
    self.model_path = os.path.join(self.dir.name, "model.tflite")
    with open(self.model_path, "wb") as file:
      file.write(build_test_model())

  def tearDown(self):
    self.dir.cleanup()

  def test_writes_to_stdout(self):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      status = propose_spec.main(["propose_spec", self.model_path])
    self.assertEqual(status, 0)
    self.assertEqual(entries(stdout.getvalue()), {(0, 1): (2, None)})

  def test_writes_to_output(self):
    output = os.path.join(self.dir.name, "spec.yaml")
    status = propose_spec.main(
      [
        "propose_spec",
        self.model_path,
        "--min_savings",
        "0",
        "--output",
        output,
      ]
    )
    self.assertEqual(status, 0)
    with open(output) as file:
      self.assertEqual(
        entries(file.read()), {(0, 1): (2, None), (0, 2): (5, None)}
      )

  def test_negative_floor_refused(self):
    with contextlib.redirect_stderr(io.StringIO()):
      with self.assertRaises(SystemExit):
        propose_spec.main(
          ["propose_spec", self.model_path, "--min_savings", "-1"]
        )


if __name__ == "__main__":
  unittest.main()
