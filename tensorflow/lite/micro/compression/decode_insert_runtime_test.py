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
"""Runtime tests for DECODE outputs crossing subgraph boundaries.

These tests build multi-subgraph models containing WHILE, compress selected
constant tensors with the LUT compressor, insert DECODE operators with
decode_insert, and run the result on the TFLM interpreter. They exercise the
two directions in which a DECODE output can cross a subgraph boundary:

1. Out of a callee subgraph: a compressed constant listed in a callee
   subgraph's output list, decoded by an appended DECODE, and copied to the
   calling operator's outputs by the runtime.

2. Into a callee subgraph: a compressed constant consumed by a WHILE
   operator, whose decoded value the runtime copies into the cond and body
   subgraph inputs. WHILE reads its inputs again after invoking the cond
   subgraph, so a DECODE inside cond that shares alternate decompression
   memory with the caller's DECODE can overwrite the value between reads.

Each case runs both with the arena memory planner and with alternate
decompression memory, since the two allocate DECODE outputs differently:
the planner gives each output a distinct, lifetime-managed buffer, while
alternate memory restarts at the same base address for every DECODE.
"""

import numpy as np
import unittest

from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.micro.compression import decode_insert
from tflite_micro.tensorflow.lite.micro.compression import lut
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite

_SHAPE = (4, )
_ARENA_SIZE = 65536
_ALT_MEMORY_SIZE = 1024


def _while_operator(cond_subgraph_idx, body_subgraph_idx, inputs, outputs):
  """Create a WHILE operator with its subgraph indices.

  model_editor has no public API for builtin options, so set them on the
  backing OperatorT directly.
  """
  op = model_editor.Operator(
      opcode=tflite.BuiltinOperator.WHILE,
      inputs=inputs,
      outputs=outputs,
  )
  options = tflite.WhileOptionsT()
  options.condSubgraphIndex = cond_subgraph_idx
  options.bodySubgraphIndex = body_subgraph_idx
  op._fb.builtinOptionsType = tflite.BuiltinOptions.WhileOptions
  op._fb.builtinOptions = options
  return op


def _float_tensor(name, data=None):
  return model_editor.Tensor(
      shape=_SHAPE,
      dtype=tflite.TensorType.FLOAT32,
      data=data,
      name=name,
  )


def _cond_subgraph(threshold_values):
  """Build a cond subgraph computing LESS(input, threshold_constant)."""
  c_in = _float_tensor("cond_in")
  threshold = _float_tensor("threshold",
                            np.array(threshold_values, dtype=np.float32))
  cond_out = model_editor.Tensor(
      shape=_SHAPE,
      dtype=tflite.TensorType.BOOL,
      name="cond_out",
  )
  return model_editor.Subgraph(
      tensors=[threshold],
      operators=[
          model_editor.Operator(
              opcode=tflite.BuiltinOperator.LESS,
              inputs=[c_in, threshold],
              outputs=[cond_out],
          )
      ],
      inputs=[c_in],
      outputs=[cond_out],
  )


def _build_body_output_model():
  """Model whose WHILE body subgraph outputs a constant.

  Subgraph 0 feeds input x into WHILE. The cond subgraph tests x < 5. The
  body subgraph has no operators; its sole output is the constant K =
  [7, 8, 7, 8]. With x = 0: cond is true, the body replaces x with K, cond
  is then false (7 < 5), and the model outputs K.

  The tensor at coordinates (2, 0) is K, the body output constant. The
  tensor at (1, 0) is the cond threshold.
  """
  x0 = _float_tensor("x0")
  y0 = _float_tensor("y0")
  sg0 = model_editor.Subgraph(
      operators=[_while_operator(1, 2, [x0], [y0])],
      inputs=[x0],
      outputs=[y0],
  )

  sg1 = _cond_subgraph([5.0, 6.0, 5.0, 6.0])

  b_in = _float_tensor("body_in")
  k = _float_tensor("k", np.array([7.0, 8.0, 7.0, 8.0], dtype=np.float32))
  sg2 = model_editor.Subgraph(
      tensors=[k],
      operators=[],
      inputs=[b_in],
      outputs=[k],
  )

  model = model_editor.Model(subgraphs=[sg0, sg1, sg2])
  model._fb.version = 3
  return model


def _build_while_input_model():
  """Model whose WHILE operator consumes a constant as its input.

  Subgraph 0 feeds the constant INIT = [10, 20, 10, 20] into WHILE as the
  initial loop value. The cond subgraph tests x < 3, false from the start,
  so the loop body (ADD 1) never runs and the model outputs INIT unchanged.

  The tensor at coordinates (0, 0) is INIT, the WHILE input constant. The
  tensor at (1, 0) is the cond threshold.
  """
  init = _float_tensor("init",
                       np.array([10.0, 20.0, 10.0, 20.0], dtype=np.float32))
  y0 = _float_tensor("y0")
  sg0 = model_editor.Subgraph(
      tensors=[init],
      operators=[_while_operator(1, 2, [init], [y0])],
      inputs=[],
      outputs=[y0],
  )

  sg1 = _cond_subgraph([3.0, 4.0, 3.0, 4.0])

  b_in = _float_tensor("body_in")
  one = _float_tensor("one", np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
  b_out = _float_tensor("body_out")
  sg2 = model_editor.Subgraph(
      operators=[
          model_editor.Operator(
              opcode=tflite.BuiltinOperator.ADD,
              inputs=[b_in, one],
              outputs=[b_out],
          )
      ],
      inputs=[b_in],
      outputs=[b_out],
  )

  model = model_editor.Model(subgraphs=[sg0, sg1, sg2])
  model._fb.version = 3
  return model


def _compress_and_insert(model, coordinates):
  """LUT-compress the tensors at (subgraph, tensor) coordinates and insert
  DECODE operators for them."""
  compressor_plugin = lut.LutCompressor()
  method = spec.LookUpTableCompression(index_bitwidth=1)
  results = {}
  for sg_idx, tensor_idx in coordinates:
    tensor = model.subgraphs[sg_idx].tensors[tensor_idx]
    results[(sg_idx, tensor_idx)] = compressor_plugin.compress(tensor, method)
  decode_insert.insert_decode_operators(model, results)


def _run(model, x0=None, alt_memory_size=0):
  """Build the model and run one inference on the TFLM interpreter."""
  flatbuffer = bytes(model.build())
  interpreter = runtime.Interpreter.from_bytes(
      flatbuffer,
      custom_op_registerers=[],
      arena_size=_ARENA_SIZE,
      alt_decompression_memory_size=alt_memory_size,
  )
  if x0 is not None:
    interpreter.set_input(x0, 0)
  interpreter.invoke()
  return interpreter.get_output(0)


class TestDecodeOutOfSubgraph(unittest.TestCase):
  """A DECODE output listed as a callee subgraph output.

  The calling operator copies callee outputs to its own outputs immediately
  after the callee returns, before any other subgraph (and thus any other
  DECODE) runs, so the decoded values must survive in both memory modes.
  """

  X0 = np.zeros(_SHAPE, dtype=np.float32)
  EXPECTED = np.array([7.0, 8.0, 7.0, 8.0], dtype=np.float32)

  def test_arena(self):
    model = _build_body_output_model()
    _compress_and_insert(model, [(2, 0)])
    output = _run(model, x0=self.X0)
    np.testing.assert_array_equal(output, self.EXPECTED)

  def test_alt_memory(self):
    model = _build_body_output_model()
    _compress_and_insert(model, [(2, 0)])
    output = _run(model, x0=self.X0, alt_memory_size=_ALT_MEMORY_SIZE)
    np.testing.assert_array_equal(output, self.EXPECTED)

  def test_alt_memory_with_decode_in_cond(self):
    # The cond threshold is also compressed, so a DECODE inside cond
    # overwrites the shared alternate memory after the body's decoded
    # output was copied out. The copy must have preserved the values.
    model = _build_body_output_model()
    _compress_and_insert(model, [(2, 0), (1, 0)])
    output = _run(model, x0=self.X0, alt_memory_size=_ALT_MEMORY_SIZE)
    np.testing.assert_array_equal(output, self.EXPECTED)


class TestDecodeIntoSubgraph(unittest.TestCase):
  """A DECODE output consumed as a WHILE operator input.

  WHILE reads its inputs once to seed the cond subgraph, then again after
  cond returns, to seed the body subgraph and initialize its outputs. A
  DECODE inside cond that shares alternate decompression memory with the
  DECODE feeding WHILE overwrites the value between those reads.
  """

  EXPECTED = np.array([10.0, 20.0, 10.0, 20.0], dtype=np.float32)

  def test_arena(self):
    model = _build_while_input_model()
    _compress_and_insert(model, [(0, 0), (1, 0)])
    output = _run(model)
    np.testing.assert_array_equal(output, self.EXPECTED)

  def test_alt_memory(self):
    model = _build_while_input_model()
    _compress_and_insert(model, [(0, 0)])
    output = _run(model, alt_memory_size=_ALT_MEMORY_SIZE)
    np.testing.assert_array_equal(output, self.EXPECTED)

  def test_alt_memory_with_decode_in_cond(self):
    model = _build_while_input_model()
    _compress_and_insert(model, [(0, 0), (1, 0)])
    output = _run(model, alt_memory_size=_ALT_MEMORY_SIZE)
    np.testing.assert_array_equal(output, self.EXPECTED)


if __name__ == "__main__":
  unittest.main()
