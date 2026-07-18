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
"""Unit tests for DECODE operator insertion."""

import warnings

import numpy as np
import unittest

from tflite_micro.tensorflow.lite.micro.compression import compressor
from tflite_micro.tensorflow.lite.micro.compression import decode
from tflite_micro.tensorflow.lite.micro.compression import decode_insert
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


def _build_simple_fc_model():
  """Build a simple model with one FC operator and compressible weights."""
  # yapf: disable
  weights = model_editor.Tensor(
      shape=(4, 4),
      dtype=tflite.TensorType.INT8,
      data=np.array([[1, 2, 1, 2],
                     [3, 4, 3, 4],
                     [1, 2, 1, 2],
                     [3, 4, 3, 4]], dtype=np.int8),
      name="weights",
      quantization=model_editor.Quantization(scales=0.5, zero_points=0),
  )
  # yapf: enable
  input_t = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="input",
  )
  output_t = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="output",
  )

  model = model_editor.Model(subgraphs=[
      model_editor.Subgraph(
          tensors=[weights],
          operators=[
              model_editor.Operator(
                  opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[input_t, weights],
                  outputs=[output_t],
              )
          ],
          inputs=[input_t],
          outputs=[output_t],
      )
  ])
  return model


def _build_shared_weights_model():
  """Build model where one tensor is used by multiple operators."""
  weights = model_editor.Tensor(
      shape=(4, 4),
      dtype=tflite.TensorType.INT8,
      data=np.ones((4, 4), dtype=np.int8),
      name="shared_weights",
      quantization=model_editor.Quantization(scales=0.5, zero_points=0),
  )
  input1 = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="input1",
  )
  input2 = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="input2",
  )
  output1 = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="output1",
  )
  output2 = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="output2",
  )

  model = model_editor.Model(subgraphs=[
      model_editor.Subgraph(
          tensors=[weights],
          operators=[
              model_editor.Operator(
                  opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[input1, weights],
                  outputs=[output1],
              ),
              model_editor.Operator(
                  opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[input2, weights],
                  outputs=[output2],
              ),
          ],
          inputs=[input1, input2],
          outputs=[output1, output2],
      )
  ])
  return model


def _build_output_constant_model():
  """Build a model where a compressed constant is a subgraph output."""
  table = model_editor.Tensor(
      shape=(4, 4),
      dtype=tflite.TensorType.INT8,
      data=np.ones((4, 4), dtype=np.int8),
      name="table",
      quantization=model_editor.Quantization(scales=0.5, zero_points=0),
  )
  weights = model_editor.Tensor(
      shape=(4, 4),
      dtype=tflite.TensorType.INT8,
      data=np.ones((4, 4), dtype=np.int8),
      name="weights",
      quantization=model_editor.Quantization(scales=0.5, zero_points=0),
  )
  input_t = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="input",
  )
  output_t = model_editor.Tensor(
      shape=(1, 4),
      dtype=tflite.TensorType.INT8,
      name="output",
  )

  model = model_editor.Model(subgraphs=[
      model_editor.Subgraph(
          tensors=[table],
          operators=[
              model_editor.Operator(
                  opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                  inputs=[input_t, weights],
                  outputs=[output_t],
              )
          ],
          inputs=[input_t],
          outputs=[output_t, table],
      )
  ])
  return model


def _build_shared_buffer_model(subgraph_count):
  """Build subgraphs whose weights tensors all share one Buffer.

  Models the TfLite converter's deduplication of identical constants:
  distinct tensors, in the same or different subgraphs, backed by a
  single Buffer.
  """
  shared = model_editor.Buffer(data=np.ones((4, 4), dtype=np.int8).tobytes())
  subgraphs = []
  for i in range(subgraph_count):
    weights = model_editor.Tensor(
        shape=(4, 4),
        dtype=tflite.TensorType.INT8,
        buffer=shared,
        name=f"weights{i}",
        quantization=model_editor.Quantization(scales=0.5, zero_points=0),
    )
    input_t = model_editor.Tensor(
        shape=(1, 4),
        dtype=tflite.TensorType.INT8,
        name=f"input{i}",
    )
    output_t = model_editor.Tensor(
        shape=(1, 4),
        dtype=tflite.TensorType.INT8,
        name=f"output{i}",
    )
    subgraphs.append(
        model_editor.Subgraph(
            tensors=[weights],
            operators=[
                model_editor.Operator(
                    opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                    inputs=[input_t, weights],
                    outputs=[output_t],
                )
            ],
            inputs=[input_t],
            outputs=[output_t],
        ))
  return model_editor.Model(subgraphs=subgraphs)


def _make_dummy_ancillary_data() -> bytes:
  """Create dummy ancillary data for testing."""
  dcm = decode.DecodeCommonMetadata(
      decode_type=decode.DecodeType.LUT,
      user_data=b'\x01\x04\x10' + b'\x00' * 9,  # lut_version, bitwidth, stride
  )
  value_tables = bytes([1, 2, 3, 4] + [0] * 12)  # 16-byte padded table
  return dcm.to_bytes() + value_tables


class TestDecodeInsertion(unittest.TestCase):
  """Tests for insert_decode_operators function."""

  def test_insert_single_decode_operator(self):
    """DECODE operator inserted before FC that uses compressed weights."""
    model = _build_simple_fc_model()

    # Create compression result
    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    # Insert DECODE operators
    decode_insert.insert_decode_operators(model, compression_results)

    sg = model.subgraphs[0]

    # Should have 2 operators: DECODE then FC
    self.assertEqual(len(sg.operators), 2)
    self.assertEqual(sg.operators[0].opcode, tflite.BuiltinOperator.CUSTOM)
    self.assertEqual(sg.operators[0].custom_code,
                     decode_insert.DECODE_CUSTOM_OP_NAME)
    self.assertEqual(sg.operators[1].opcode,
                     tflite.BuiltinOperator.FULLY_CONNECTED)

  def test_decode_inputs_structure(self):
    """DECODE operator has correct inputs: encoded tensor + ancillary."""
    model = _build_simple_fc_model()
    weights_tensor = model.subgraphs[0].tensor_by_name("weights")

    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    decode_insert.insert_decode_operators(model, compression_results)

    decode_op = model.subgraphs[0].operators[0]

    # DECODE has 2 inputs
    self.assertEqual(len(decode_op.inputs), 2)
    # First input is the encoded tensor (original weights)
    self.assertIs(decode_op.inputs[0], weights_tensor)
    # Second input is ancillary tensor
    self.assertEqual(decode_op.inputs[1].dtype, tflite.TensorType.UINT8)

  def test_decode_output_structure(self):
    """DECODE operator output is a data-less copy of the original."""
    model = _build_simple_fc_model()
    weights_tensor = model.subgraphs[0].tensor_by_name("weights")

    # Snapshot what the output must look like before insertion rewrites
    # the original into the encoded tensor: a copy of the original,
    # renamed, with no data.
    expected = weights_tensor.copy(name="weights_decoded")
    expected.buffer = None

    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    decode_insert.insert_decode_operators(model, compression_results)

    output = model.subgraphs[0].operators[0].outputs[0]

    # Output has no data; DECODE produces the values at runtime
    self.assertIsNone(output.buffer)
    self.assertIsNone(output.array)

    # Output matches the expected copy in every field, present or
    # future; the new name and cleared buffer are part of the
    # expectation, not exclusions
    self.assertTrue(output.equal(expected))

  def test_consumer_rewired_to_decode_output(self):
    """FC operator input rewired to use DECODE output."""
    model = _build_simple_fc_model()
    weights_tensor = model.subgraphs[0].tensor_by_name("weights")

    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    decode_insert.insert_decode_operators(model, compression_results)

    decode_op = model.subgraphs[0].operators[0]
    fc_op = model.subgraphs[0].operators[1]

    # FC's second input (weights) should now be DECODE's output
    self.assertIs(fc_op.inputs[1], decode_op.outputs[0])
    # Original weights tensor should NOT be in FC inputs
    self.assertNotIn(weights_tensor, fc_op.inputs)

  def test_shared_tensor_decode_per_consumer(self):
    """Tensor used by multiple ops gets separate DECODE for each consumer."""
    model = _build_shared_weights_model()

    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    decode_insert.insert_decode_operators(model, compression_results)

    sg = model.subgraphs[0]

    # Should have 4 operators: 2 DECODEs + 2 FCs (DECODE before each FC)
    self.assertEqual(len(sg.operators), 4)
    self.assertEqual(sg.operators[0].opcode, tflite.BuiltinOperator.CUSTOM)
    self.assertEqual(sg.operators[1].opcode,
                     tflite.BuiltinOperator.FULLY_CONNECTED)
    self.assertEqual(sg.operators[2].opcode, tflite.BuiltinOperator.CUSTOM)
    self.assertEqual(sg.operators[3].opcode,
                     tflite.BuiltinOperator.FULLY_CONNECTED)

    decode_op1 = sg.operators[0]
    fc_op1 = sg.operators[1]
    decode_op2 = sg.operators[2]
    fc_op2 = sg.operators[3]

    # Each FC should use its own DECODE's output
    self.assertIs(fc_op1.inputs[1], decode_op1.outputs[0])
    self.assertIs(fc_op2.inputs[1], decode_op2.outputs[0])
    # The two DECODEs should have different outputs
    self.assertIsNot(decode_op1.outputs[0], decode_op2.outputs[0])
    # The two DECODEs should share the same ancillary tensor
    self.assertIs(decode_op1.inputs[1], decode_op2.inputs[1])

  def test_ancillary_buffer_shared_across_subgraphs(self):
    """Tensors sharing a Buffer get ancillary tensors sharing a Buffer."""
    model = _build_shared_buffer_model(subgraph_count=2)

    ancillary_data = _make_dummy_ancillary_data()
    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=ancillary_data,
        ),
        (1, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=ancillary_data,
        ),
    }

    decode_insert.insert_decode_operators(model, compression_results)

    decode0 = model.subgraphs[0].operators[0]
    decode1 = model.subgraphs[1].operators[0]
    encoded0, ancillary0 = decode0.inputs[0], decode0.inputs[1]
    encoded1, ancillary1 = decode1.inputs[0], decode1.inputs[1]

    # Tensors are per-subgraph; buffers are shared across subgraphs
    self.assertIsNot(ancillary0, ancillary1)
    self.assertIs(ancillary0.buffer, ancillary1.buffer)

    # The encoded tensors keep sharing their original Buffer
    self.assertIsNot(encoded0, encoded1)
    self.assertIs(encoded0.buffer, encoded1.buffer)

  def test_ancillary_tensor_shared_within_subgraph(self):
    """Aliased tensors in one subgraph share one ancillary tensor."""
    model = _build_shared_buffer_model(subgraph_count=1)
    sg = model.subgraphs[0]

    # Add a second tensor aliasing the weights buffer, with its own
    # consumer; copy() shares the buffer, as converter dedup does
    weights = sg.tensor_by_name("weights0")
    alias = weights.copy(name="alias")
    sg.tensors.append(alias)
    input_t = model_editor.Tensor(
        shape=(1, 4),
        dtype=tflite.TensorType.INT8,
        name="input_alias",
    )
    output_t = model_editor.Tensor(
        shape=(1, 4),
        dtype=tflite.TensorType.INT8,
        name="output_alias",
    )
    sg.operators.append(
        model_editor.Operator(
            opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
            inputs=[input_t, alias],
            outputs=[output_t],
        ))

    ancillary_data = _make_dummy_ancillary_data()
    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=ancillary_data,
        ),
        (0, 1):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=ancillary_data,
        ),
    }

    decode_insert.insert_decode_operators(model, compression_results)

    # One DECODE before each consumer, sharing one ancillary tensor
    decodes = [
        op for op in sg.operators
        if op.custom_code == decode_insert.DECODE_CUSTOM_OP_NAME
    ]
    self.assertEqual(len(decodes), 2)
    self.assertIs(decodes[0].inputs[1], decodes[1].inputs[1])

  def test_output_constant_gets_decode(self):
    """Compressed tensor in subgraph outputs gets DECODE appended last."""
    model = _build_output_constant_model()
    sg = model.subgraphs[0]
    table = sg.tensor_by_name("table")
    original_first_output = sg.outputs[0]

    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    decode_insert.insert_decode_operators(model, compression_results)

    # DECODE appended after the FC operator
    self.assertEqual(len(sg.operators), 2)
    decode_op = sg.operators[1]
    self.assertEqual(decode_op.opcode, tflite.BuiltinOperator.CUSTOM)
    self.assertEqual(decode_op.custom_code,
                     decode_insert.DECODE_CUSTOM_OP_NAME)
    self.assertIs(decode_op.inputs[0], table)

    # Output list entry rewired to the decoded tensor; other entry untouched
    self.assertIs(sg.outputs[0], original_first_output)
    self.assertIs(sg.outputs[1], decode_op.outputs[0])

    # Original tensor rewritten to encoded data
    self.assertEqual(table.dtype, tflite.TensorType.UINT8)

  def test_consumed_and_output_tensor(self):
    """Tensor both consumed and a subgraph output gets both DECODEs."""
    model = _build_simple_fc_model()
    sg = model.subgraphs[0]
    weights = sg.tensor_by_name("weights")
    fc_output = sg.operators[0].outputs[0]
    sg.outputs = [fc_output, weights]

    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    decode_insert.insert_decode_operators(model, compression_results)

    # Consumer DECODE before FC, output DECODE appended last
    self.assertEqual(len(sg.operators), 3)
    consumer_decode = sg.operators[0]
    fc_op = sg.operators[1]
    output_decode = sg.operators[2]
    self.assertEqual(consumer_decode.opcode, tflite.BuiltinOperator.CUSTOM)
    self.assertEqual(fc_op.opcode, tflite.BuiltinOperator.FULLY_CONNECTED)
    self.assertEqual(output_decode.opcode, tflite.BuiltinOperator.CUSTOM)

    # Each reader gets its own decoded tensor
    self.assertIs(fc_op.inputs[1], consumer_decode.outputs[0])
    self.assertIs(sg.outputs[1], output_decode.outputs[0])
    self.assertIsNot(consumer_decode.outputs[0], output_decode.outputs[0])

    # Both DECODEs share the ancillary tensor
    self.assertIs(consumer_decode.inputs[1], output_decode.inputs[1])

  def test_multiple_input_tensors_share_one_decode(self):
    """All compressed tensors of one consumer are decoded by one DECODE."""
    weights1 = model_editor.Tensor(
        shape=(4, 4),
        dtype=tflite.TensorType.INT8,
        data=np.ones((4, 4), dtype=np.int8),
        name="weights1",
        quantization=model_editor.Quantization(scales=0.5, zero_points=0),
    )
    weights2 = model_editor.Tensor(
        shape=(1, 4),
        dtype=tflite.TensorType.INT8,
        data=np.ones((1, 4), dtype=np.int8),
        name="weights2",
        quantization=model_editor.Quantization(scales=0.5, zero_points=0),
    )
    input_t = model_editor.Tensor(
        shape=(1, 4),
        dtype=tflite.TensorType.INT8,
        name="input",
    )
    output_t = model_editor.Tensor(
        shape=(1, 4),
        dtype=tflite.TensorType.INT8,
        name="output",
    )

    model = model_editor.Model(subgraphs=[
        model_editor.Subgraph(
            tensors=[weights1, weights2],
            operators=[
                model_editor.Operator(
                    opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                    inputs=[input_t, weights1, weights2],
                    outputs=[output_t],
                )
            ],
        )
    ])
    sg = model.subgraphs[0]

    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        ),
        (0, 1):
        compressor.CompressionResult(
            encoded_data=b'\x01\x01',
            ancillary_data=_make_dummy_ancillary_data(),
        ),
    }

    decode_insert.insert_decode_operators(model, compression_results)

    # One DECODE with two encoded/ancillary pairs, inserted before the FC
    self.assertEqual(len(sg.operators), 2)
    decode_op = sg.operators[0]
    fc_op = sg.operators[1]
    self.assertEqual(decode_op.opcode, tflite.BuiltinOperator.CUSTOM)
    self.assertEqual(fc_op.opcode, tflite.BuiltinOperator.FULLY_CONNECTED)
    self.assertEqual(len(decode_op.inputs), 4)
    self.assertEqual(len(decode_op.outputs), 2)
    self.assertIs(decode_op.inputs[0], weights1)
    self.assertIs(decode_op.inputs[2], weights2)

    # The consumer reads each decoded tensor from its original position
    self.assertIs(fc_op.inputs[1], decode_op.outputs[0])
    self.assertIs(fc_op.inputs[2], decode_op.outputs[1])

  def test_multiple_output_tensors_share_one_decode(self):
    """All compressed subgraph outputs are decoded by a single DECODE."""
    table1 = model_editor.Tensor(
        shape=(4, 4),
        dtype=tflite.TensorType.INT8,
        data=np.ones((4, 4), dtype=np.int8),
        name="table1",
        quantization=model_editor.Quantization(scales=0.5, zero_points=0),
    )
    table2 = model_editor.Tensor(
        shape=(2, 2),
        dtype=tflite.TensorType.INT8,
        data=np.ones((2, 2), dtype=np.int8),
        name="table2",
        quantization=model_editor.Quantization(scales=0.5, zero_points=0),
    )
    output_t = model_editor.Tensor(
        shape=(1, 4),
        dtype=tflite.TensorType.INT8,
        name="output",
    )

    model = model_editor.Model(subgraphs=[
        model_editor.Subgraph(
            tensors=[table1, table2],
            operators=[],
            outputs=[table1, output_t, table2],
        )
    ])
    sg = model.subgraphs[0]

    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        ),
        (0, 1):
        compressor.CompressionResult(
            encoded_data=b'\x01\x01',
            ancillary_data=_make_dummy_ancillary_data(),
        ),
    }

    decode_insert.insert_decode_operators(model, compression_results)

    # One DECODE with two encoded/ancillary pairs and two outputs
    self.assertEqual(len(sg.operators), 1)
    decode_op = sg.operators[0]
    self.assertEqual(len(decode_op.inputs), 4)
    self.assertEqual(len(decode_op.outputs), 2)
    self.assertIs(decode_op.inputs[0], table1)
    self.assertIs(decode_op.inputs[2], table2)

    # Output list rewired in place; uncompressed entry untouched
    self.assertIs(sg.outputs[0], decode_op.outputs[0])
    self.assertIs(sg.outputs[1], output_t)
    self.assertIs(sg.outputs[2], decode_op.outputs[1])

  def test_ancillary_tensor_contains_dcm(self):
    """Ancillary tensor data contains valid DCM header."""
    model = _build_simple_fc_model()

    ancillary_data = _make_dummy_ancillary_data()
    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=ancillary_data,
        )
    }

    decode_insert.insert_decode_operators(model, compression_results)

    decode_op = model.subgraphs[0].operators[0]
    ancillary_tensor = decode_op.inputs[1]

    # Ancillary tensor data should match what we provided
    self.assertEqual(bytes(ancillary_tensor.array), ancillary_data)

    # Verify DCM header
    dcm_bytes = ancillary_tensor.array[:16]
    self.assertEqual(dcm_bytes[0], 0)  # decode_type = LUT
    self.assertEqual(dcm_bytes[1], 1)  # DCM version

  def test_no_consumers_no_decode(self):
    """Tensor with no consumers gets no DECODE operator and emits warning."""
    # Create model where compressed tensor is not used as input
    unused_tensor = model_editor.Tensor(
        shape=(4, 4),
        dtype=tflite.TensorType.INT8,
        data=np.ones((4, 4), dtype=np.int8),
        name="unused",
        quantization=model_editor.Quantization(scales=0.5, zero_points=0),
    )
    input_t = model_editor.Tensor(
        shape=(1, 4),
        dtype=tflite.TensorType.INT8,
        name="input",
    )
    output_t = model_editor.Tensor(
        shape=(1, 4),
        dtype=tflite.TensorType.INT8,
        name="output",
    )
    other_weights = model_editor.Tensor(
        shape=(4, 4),
        dtype=tflite.TensorType.INT8,
        data=np.ones((4, 4), dtype=np.int8),
        name="other_weights",
        quantization=model_editor.Quantization(scales=0.5, zero_points=0),
    )

    model = model_editor.Model(subgraphs=[
        model_editor.Subgraph(
            tensors=[unused_tensor, other_weights],
            operators=[
                model_editor.Operator(
                    opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                    inputs=[input_t, other_weights],
                    outputs=[output_t],
                )
            ],
        )
    ])

    # Compress the unused tensor
    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      decode_insert.insert_decode_operators(model, compression_results)

      # Should emit a warning about no consumers
      self.assertEqual(len(w), 1)
      self.assertIn("no consumers", str(w[0].message))
      self.assertIn("unused", str(w[0].message))

    # Should still have just 1 operator (no DECODE inserted)
    self.assertEqual(len(model.subgraphs[0].operators), 1)

  def test_tensor_naming(self):
    """Output and ancillary tensors get appropriate names."""
    model = _build_simple_fc_model()

    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    decode_insert.insert_decode_operators(model, compression_results)

    decode_op = model.subgraphs[0].operators[0]
    ancillary = decode_op.inputs[1]
    output = decode_op.outputs[0]

    self.assertEqual(ancillary.name, "weights_ancillary")
    self.assertEqual(output.name, "weights_decoded")

  def test_encoded_tensor_rewritten(self):
    """Compressed tensor is rewritten with encoded data, UINT8 type, no quant."""
    model = _build_simple_fc_model()
    weights_tensor = model.subgraphs[0].tensor_by_name("weights")

    encoded_data = b'\xAB\xCD\xEF'
    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=encoded_data,
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    decode_insert.insert_decode_operators(model, compression_results)

    # Original tensor should be rewritten
    self.assertEqual(weights_tensor.shape, (len(encoded_data), ))
    self.assertEqual(weights_tensor.dtype, tflite.TensorType.UINT8)
    self.assertIsNone(weights_tensor.quantization)
    self.assertEqual(weights_tensor.buffer.data, encoded_data)


if __name__ == "__main__":
  unittest.main()
