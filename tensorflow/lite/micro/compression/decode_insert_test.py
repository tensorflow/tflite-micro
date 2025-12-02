# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np
import tensorflow as tf

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
      )
  ])
  return model


def _make_dummy_ancillary_data() -> bytes:
  """Create dummy ancillary data for testing."""
  dcm = decode.DecodeCommonMetadata(
      decode_type=decode.DecodeType.LUT,
      user_data=b'\x01\x04\x10' + b'\x00' * 9,  # lut_version, bitwidth, stride
  )
  value_tables = bytes([1, 2, 3, 4] + [0] * 12)  # 16-byte padded table
  return dcm.to_bytes() + value_tables


class TestDecodeInsertion(tf.test.TestCase):
  """Tests for insert_decode_operators function."""

  def test_insert_single_decode_operator(self):
    """DECODE operator inserted before FC that uses compressed weights."""
    model = _build_simple_fc_model()
    weights_tensor = model.subgraphs[0].tensors[0]

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
    weights_tensor = model.subgraphs[0].tensors[0]

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
    """DECODE operator output has correct shape and dtype."""
    model = _build_simple_fc_model()
    weights_tensor = model.subgraphs[0].tensors[0]

    compression_results = {
        (0, 0):
        compressor.CompressionResult(
            encoded_data=b'\x00\x00',
            ancillary_data=_make_dummy_ancillary_data(),
        )
    }

    decode_insert.insert_decode_operators(model, compression_results)

    decode_op = model.subgraphs[0].operators[0]
    output = decode_op.outputs[0]

    # Output matches original tensor shape and dtype
    self.assertEqual(output.shape, weights_tensor.shape)
    self.assertEqual(output.dtype, weights_tensor.dtype)

  def test_consumer_rewired_to_decode_output(self):
    """FC operator input rewired to use DECODE output."""
    model = _build_simple_fc_model()
    weights_tensor = model.subgraphs[0].tensors[0]

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
    weights_tensor = model.subgraphs[0].tensors[0]

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
    """Tensor with no consumers gets no DECODE operator."""
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

    decode_insert.insert_decode_operators(model, compression_results)

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


class TestHelperFunctions(tf.test.TestCase):
  """Tests for internal helper functions."""

  def test_find_tensor_consumers(self):
    """_find_tensor_consumers finds all ops using a tensor."""
    model = _build_shared_weights_model()
    sg = model.subgraphs[0]
    weights = sg.tensors[0]

    consumers = decode_insert._find_tensor_consumers(sg, weights)

    self.assertEqual(len(consumers), 2)

  def test_find_earliest_consumer_position(self):
    """_find_earliest_consumer_position returns minimum position."""
    model = _build_shared_weights_model()
    sg = model.subgraphs[0]

    # Both operators are consumers, first is at position 0
    pos = decode_insert._find_earliest_consumer_position(sg, sg.operators)
    self.assertEqual(pos, 0)

    # Just the second operator
    pos = decode_insert._find_earliest_consumer_position(sg, [sg.operators[1]])
    self.assertEqual(pos, 1)


if __name__ == "__main__":
  tf.test.main()
