# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for the compression system."""

import warnings

import numpy as np
import tensorflow as tf

from tflite_micro.tensorflow.lite.micro.compression import compress
from tflite_micro.tensorflow.lite.micro.compression import compressor
from tflite_micro.tensorflow.lite.micro.compression import decode_insert
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


def _build_test_model():
  """Build test model using model_editor API."""
  from tflite_micro.tensorflow.lite.micro.compression.model_editor import (
      Model, Subgraph, Tensor, Operator, Quantization)

  # Pre-declare tensors with stable indices for compression specs
  t0 = Tensor(shape=(16, 1),
              dtype=tflite.TensorType.UINT8,
              data=np.array(range(16), dtype="<u1"),
              name="tensor0",
              quantization=Quantization(scales=1, zero_points=0))
  t1 = Tensor(shape=(16, 1),
              dtype=tflite.TensorType.INT8,
              data=np.array(range(-16, 0), dtype="<i1"),
              name="tensor1",
              quantization=Quantization(scales=1, zero_points=0))
  t2 = Tensor(shape=(16, 1),
              dtype=tflite.TensorType.INT16,
              data=np.array(range(-1616, -1600), dtype="<i2"),
              name="tensor2",
              quantization=Quantization(scales=1, zero_points=0))
  t3 = Tensor(shape=(16, 1),
              dtype=tflite.TensorType.INT32,
              data=np.array(range(-160_016, -160_000), dtype="<i4"),
              name="tensor3",
              quantization=Quantization(scales=1, zero_points=0))
  t4 = Tensor(shape=(16, 1),
              dtype=tflite.TensorType.INT32,
              data=np.array(range(16), dtype="<i4"),
              name="tensor4_uncompressed",
              quantization=Quantization(scales=1, zero_points=0))
  # yapf: disable
  t5 = Tensor(
      shape=(4, 5),
      dtype=tflite.TensorType.INT16,
      data=np.array((( 1,  5,  9, 13, 17),
                     ( 2,  6, 10, 14, 18),
                     ( 3,  7, 11, 15, 19),
                     ( 4,  8, 12, 16, 20)), dtype="<i2"),
      name="tensor5_perchannel",
      quantization=Quantization(
          scales=[1, 1, 1, 1, 1], zero_points=[0, 0, 0, 0, 0], axis=1))
  t6 = Tensor(
      shape=(5, 4),
      dtype=tflite.TensorType.INT16,
      data=np.array((( 1,  2,  3,  4),
                     ( 5,  6,  7,  8),
                     ( 9, 10, 11, 12),
                     (13, 14, 15, 16),
                     (17, 18, 19, 20)), dtype="<i2"),
      name="tensor6_axis0",
      quantization=Quantization(
          scales=[1, 1, 1, 1, 1], zero_points=[0, 0, 0, 0, 0], axis=0))
  t7 = Tensor(
      shape=(5, 4),
      dtype=tflite.TensorType.INT16,
      data=np.array(((1, 2, 3, 4),
                     (1, 2, 3, 4),
                     (1, 2, 3, 4),
                     (1, 2, 3, 4),
                     (1, 2, 3, 4)), dtype="<i2"),
      name="tensor7_pertensor",
      quantization=Quantization(scales=1, zero_points=0))
  # yapf: enable
  t8 = Tensor(shape=(16, 1),
              dtype=tflite.TensorType.UINT8,
              data=np.array(range(16), dtype="<u1"),
              name="tensor8_no_quantization")

  # Output tensors (no data)
  out0 = Tensor(shape=(16, 1), dtype=tflite.TensorType.INT16, name="output0")
  out1 = Tensor(shape=(16, 1), dtype=tflite.TensorType.INT16, name="output1")

  model = Model(metadata={"metadata0": b""},
                subgraphs=[
                    Subgraph(tensors=[t0, t1, t2, t3, t4, t5, t6, t7, t8],
                             operators=[
                                 Operator(opcode=tflite.BuiltinOperator.ADD,
                                          inputs=[t0, t1],
                                          outputs=[out0]),
                                 Operator(opcode=tflite.BuiltinOperator.MUL,
                                          inputs=[t2, t3],
                                          outputs=[out1]),
                             ])
                ])

  return model.build()


TEST_COMPRESSION_SPEC = [
    spec.Tensor(  # spec 0
        subgraph=0,
        tensor=0,
        compression=[spec.LookUpTableCompression(index_bitwidth=4)],
    ),
    spec.Tensor(  # spec 1
        subgraph=0,
        tensor=1,
        compression=[spec.LookUpTableCompression(index_bitwidth=4)],
    ),
    spec.Tensor(  # spec 2
        subgraph=0,
        tensor=2,
        compression=[spec.LookUpTableCompression(index_bitwidth=4)],
    ),
    spec.Tensor(  # spec 3
        subgraph=0,
        tensor=3,
        compression=[spec.LookUpTableCompression(index_bitwidth=4)],
    ),

    # Tensor 4 intentionally left uncompressed
    spec.Tensor(  # spec 4
        subgraph=0,
        tensor=5,
        compression=[spec.LookUpTableCompression(index_bitwidth=2)],
    ),
    spec.Tensor(  # spec 5
        subgraph=0,
        tensor=6,
        compression=[spec.LookUpTableCompression(index_bitwidth=2)],
    ),
    spec.Tensor(  # spec 6
        subgraph=0,
        tensor=7,
        compression=[spec.LookUpTableCompression(index_bitwidth=2)],
    ),
]


class TestCompression(tf.test.TestCase):
  """Integration tests for the compress() function."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.flatbuffer = _build_test_model()
    cls.uncompressed = model_editor.read(cls.flatbuffer)

  def test_compression_produces_valid_flatbuffer(self):
    """Compressed model is a valid flatbuffer that can be read back."""
    compressed_fb = compress.compress(self.flatbuffer, TEST_COMPRESSION_SPEC)
    model = model_editor.read(compressed_fb)
    self.assertIsNotNone(model)
    self.assertEqual(len(model.subgraphs), 1)

  def test_decode_operators_inserted(self):
    """DECODE operators are inserted for compressed tensors."""
    compressed_fb = compress.compress(self.flatbuffer, TEST_COMPRESSION_SPEC)
    model = model_editor.read(compressed_fb)
    sg = model.subgraphs[0]

    # Find DECODE operators
    decode_ops = [
        op for op in sg.operators if op.opcode == tflite.BuiltinOperator.CUSTOM
        and op.custom_code == decode_insert.DECODE_CUSTOM_OP_NAME
    ]

    # Should have DECODE ops for compressed tensors that are used as inputs
    # t0, t1 used by ADD; t2, t3 used by MUL
    # t5, t6, t7 are not used as inputs in the test model
    self.assertGreater(len(decode_ops), 0)

  def test_decode_operator_structure(self):
    """DECODE operators have correct input/output structure."""
    # Build a simple model where weights are used as input
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
    fb = model.build()

    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=0,
            compression=[spec.LookUpTableCompression(index_bitwidth=4)])
    ]

    compressed_fb = compress.compress(fb, specs)
    result = model_editor.read(compressed_fb)
    sg = result.subgraphs[0]

    # Find DECODE operator
    decode_ops = [
        op for op in sg.operators if op.opcode == tflite.BuiltinOperator.CUSTOM
        and op.custom_code == decode_insert.DECODE_CUSTOM_OP_NAME
    ]
    self.assertEqual(len(decode_ops), 1)
    decode_op = decode_ops[0]

    # DECODE has 2 inputs: encoded tensor and ancillary data
    self.assertEqual(len(decode_op.inputs), 2)
    # DECODE has 1 output
    self.assertEqual(len(decode_op.outputs), 1)
    # Output has same shape as original weights
    self.assertEqual(decode_op.outputs[0].shape, (4, 4))

  def test_ancillary_data_format(self):
    """Ancillary data has correct DCM header format."""
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
    input_t = model_editor.Tensor(shape=(1, 4),
                                  dtype=tflite.TensorType.INT8,
                                  name="input")
    output_t = model_editor.Tensor(shape=(1, 4),
                                   dtype=tflite.TensorType.INT8,
                                   name="output")

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
    fb = model.build()

    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=0,
            compression=[spec.LookUpTableCompression(index_bitwidth=4)])
    ]

    compressed_fb = compress.compress(fb, specs)
    result = model_editor.read(compressed_fb)

    # Find DECODE and get ancillary tensor
    decode_op = next(op for op in result.subgraphs[0].operators
                     if op.custom_code == decode_insert.DECODE_CUSTOM_OP_NAME)
    ancillary = decode_op.inputs[1]

    # Verify DCM header
    dcm_bytes = bytes(ancillary.array[:16])
    self.assertEqual(dcm_bytes[0], 0)  # decode_type = LUT
    self.assertEqual(dcm_bytes[1], 1)  # DCM version
    self.assertEqual(dcm_bytes[4], 1)  # LUT version
    self.assertEqual(dcm_bytes[5] & 0x07, 4)  # bitwidth = 4
    self.assertEqual(dcm_bytes[6], 4)  # stride = num unique values

  def test_empty_spec_raises(self):
    """Empty compression spec is an error, not a silent no-op."""
    self.assertRaisesRegex(compressor.CompressionError, "empty",
                           lambda: compress.compress(self.flatbuffer, []))

  def test_smaller_bitwidth_raises(self):
    """Specifying LUT compression with too small a bitwidth fails."""
    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=1,
            compression=[spec.LookUpTableCompression(index_bitwidth=3)],
        ),
    ]
    self.assertRaises(compressor.CompressionError,
                      lambda: compress.compress(self.flatbuffer, specs))

  def test_larger_bitwidth_succeeds(self):
    """Specifying LUT compression with too large a bitwidth succeeds."""
    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=1,
            compression=[spec.LookUpTableCompression(index_bitwidth=5)],
        ),
    ]
    # Should not raise
    _ = compress.compress(self.flatbuffer, specs)

  def test_invalid_tensor_spec_raises(self):
    """Specifying a tensor that doesn't exist raises CompressionError."""
    specs = [
        spec.Tensor(
            subgraph=666,
            tensor=1,
            compression=[spec.LookUpTableCompression(index_bitwidth=4)],
        ),
    ]
    self.assertRaises(compressor.CompressionError,
                      lambda: compress.compress(self.flatbuffer, specs))

    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=666,
            compression=[spec.LookUpTableCompression(index_bitwidth=4)],
        ),
    ]
    self.assertRaises(compressor.CompressionError,
                      lambda: compress.compress(self.flatbuffer, specs))

  def test_no_quantization_uses_per_tensor(self):
    """Unquantized tensors compress with per-tensor compression (no error)."""
    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=8,
            compression=[spec.LookUpTableCompression(index_bitwidth=4)],
        ),
    ]
    # Should succeed - unquantized tensors use per-tensor compression
    _ = compress.compress(self.flatbuffer, specs)

  def test_huffman_compression_not_implemented(self):
    """Huffman compression raises not implemented error."""
    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=0,
            compression=[spec.HuffmanCompression()],
        ),
    ]
    self.assertRaises(compressor.CompressionError,
                      lambda: compress.compress(self.flatbuffer, specs))

  def test_pruning_compression_not_implemented(self):
    """Pruning compression raises not implemented error."""
    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=0,
            compression=[spec.PruningCompression()],
        ),
    ]
    self.assertRaises(compressor.CompressionError,
                      lambda: compress.compress(self.flatbuffer, specs))

  def test_compression_expansion_warning(self):
    """Warning emitted when compression results in expansion."""
    # Build a tiny model where compression overhead exceeds savings
    tiny_weights = model_editor.Tensor(
        shape=(2, ),
        dtype=tflite.TensorType.INT8,
        data=np.array([1, 2], dtype=np.int8),  # 2 bytes original
        name="tiny",
        quantization=model_editor.Quantization(scales=0.5, zero_points=0),
    )
    input_t = model_editor.Tensor(shape=(1, ),
                                  dtype=tflite.TensorType.INT8,
                                  name="input")
    output_t = model_editor.Tensor(shape=(1, ),
                                   dtype=tflite.TensorType.INT8,
                                   name="output")
    model = model_editor.Model(subgraphs=[
        model_editor.Subgraph(
            tensors=[tiny_weights],
            operators=[
                model_editor.Operator(
                    opcode=tflite.BuiltinOperator.FULLY_CONNECTED,
                    inputs=[input_t, tiny_weights],
                    outputs=[output_t],
                )
            ],
        )
    ])
    fb = model.build()

    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=0,
            compression=[spec.LookUpTableCompression(index_bitwidth=4)],
        )
    ]

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      compress.compress(fb, specs)

      expansion_warnings = [x for x in w if "expansion" in str(x.message)]
      self.assertEqual(len(expansion_warnings), 1)
      self.assertIn("tiny", str(expansion_warnings[0].message))


class TestPluginDispatch(tf.test.TestCase):
  """Tests for the plugin dispatch system."""

  def test_get_compressor_lut(self):
    """LUT compression method dispatches to LutCompressor."""
    method = spec.LookUpTableCompression(index_bitwidth=4)
    compressor_instance = compress._get_compressor(method)
    from tflite_micro.tensorflow.lite.micro.compression import lut
    self.assertIsInstance(compressor_instance, lut.LutCompressor)

  def test_get_compressor_huffman(self):
    """Huffman compression method dispatches to HuffmanCompressor."""
    method = spec.HuffmanCompression()
    compressor_instance = compress._get_compressor(method)
    from tflite_micro.tensorflow.lite.micro.compression import huffman
    self.assertIsInstance(compressor_instance, huffman.HuffmanCompressor)

  def test_get_compressor_pruning(self):
    """Pruning compression method dispatches to PruningCompressor."""
    method = spec.PruningCompression()
    compressor_instance = compress._get_compressor(method)
    from tflite_micro.tensorflow.lite.micro.compression import pruning
    self.assertIsInstance(compressor_instance, pruning.PruningCompressor)

  def test_get_compressor_unknown_raises(self):
    """Unknown compression method raises CompressionError."""

    class UnknownCompression(spec.CompressionMethod):
      pass

    method = UnknownCompression()
    self.assertRaises(compressor.CompressionError,
                      lambda: compress._get_compressor(method))


if __name__ == "__main__":
  tf.test.main()
