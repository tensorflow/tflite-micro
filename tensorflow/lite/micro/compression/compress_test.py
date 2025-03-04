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

import bitarray
import bitarray.util
import numpy as np
import tensorflow as tf

from tflite_micro.tensorflow.lite.micro.compression import compress
from tflite_micro.tensorflow.lite.micro.compression import metadata_py_generated as schema
from tflite_micro.tensorflow.lite.micro.compression import model_facade
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.micro.compression import test_models
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


class TestPackIndices(tf.test.TestCase):

  def test_basic_case(self):
    indices = np.array([1, 2, 3])
    bitwidth = 4
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = bytes([0b0001_0010, 0b0011_0000])
    self.assertEqual(result, expected_bytes)

  def test_single_element(self):
    indices = np.array([10])
    bitwidth = 8
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = bytes([0b0000_1010])
    self.assertEqual(result, expected_bytes)

  def test_different_bitwidth(self):
    indices = np.array([1, 2, 3])
    bitwidth = 8
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = bytes([0b0000_0001, 0b0000_0010, 0b0000_0011])
    self.assertEqual(result, expected_bytes)

  def test_large_numbers(self):
    indices = np.array([255, 128, 64])
    bitwidth = 8
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = bytes([0b1111_1111, 0b1000_0000, 0b0100_0000])
    self.assertEqual(result, expected_bytes)

  def test_multidimensional_array(self):
    indices = np.array([[1, 2], [3, 4]])
    bitwidth = 4
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = bytes([0b0001_0010, 0b0011_0100])
    self.assertEqual(result, expected_bytes)

  def test_zero_bitwidth(self):
    indices = np.array([0, 1, 2])
    bitwidth = 0
    with self.assertRaises(ValueError):
      compress._pack_indices(indices, bitwidth)

  def test_empty_array(self):
    indices = np.array([])
    bitwidth = 4
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = b""
    self.assertEqual(result, expected_bytes)

  def test_bitwidth_1(self):
    indices = np.array([1, 0, 1, 1, 0, 1])
    bitwidth = 1
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = bytes([0b101101_00])
    self.assertEqual(result, expected_bytes)

  def test_bitwidth_2(self):
    indices = np.array([1, 2, 3, 0])
    bitwidth = 2
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = bytes([0b01_10_11_00])
    self.assertEqual(result, expected_bytes)

  def test_bitwidth_3(self):
    indices = np.array([1, 3, 5, 7])
    bitwidth = 3
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = bytes([0b001_011_10, 0b1_111_0000])
    self.assertEqual(result, expected_bytes)

  def test_bitwidth_5(self):
    indices = np.array([1, 2, 16, 31])
    bitwidth = 5
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = bytes([0b00001_000, 0b10_10000_1, 0b1111_0000])
    self.assertEqual(result, expected_bytes)

  def test_bitwidth_7(self):
    indices = np.array([1, 64, 127, 32])
    bitwidth = 7
    result = compress._pack_indices(indices, bitwidth)
    expected_bytes = bytes(
        [0b0000001_1, 0b000000_11, 0b11111_010, 0b0000_0000])
    self.assertEqual(result, expected_bytes)


class TestPackLookupTables(tf.test.TestCase):

  def test_int16_positive(self):
    tables = [np.array([0x1234, 0x5678], dtype='<i2')]
    table_len = 2
    expected_output = bytes([0x34, 0x12, 0x78, 0x56])
    result = compress._pack_lookup_tables(tables, table_len)
    self.assertEqual(result, expected_output)

  def test_int16_negative(self):
    tables = [np.array([-0x1234, -0x5678], dtype='<i2')]
    table_len = 2
    # Expected output is two's complement
    expected_output = bytes([0xcc, 0xed, 0x88, 0xa9])
    result = compress._pack_lookup_tables(tables, table_len)
    self.assertEqual(result, expected_output)

  def test_float16(self):
    tables = [np.array([1.5, -2.5], dtype='<f2')]
    table_len = 2
    expected_output = bytes([0x00, 0x3e, 0x00, 0xc1])
    result = compress._pack_lookup_tables(tables, table_len)
    self.assertEqual(result, expected_output)

  def test_multiple_tables(self):
    tables = [
        np.array([0x1234, 0x5678], dtype='<i2'),
        np.array([0x6abc, 0x7ef0], dtype='<i2')
    ]
    table_len = 2
    expected_output = bytes([0x34, 0x12, 0x78, 0x56, 0xbc, 0x6a, 0xf0, 0x7e])
    result = compress._pack_lookup_tables(tables, table_len)
    self.assertEqual(result, expected_output)

  def test_int16_with_padding(self):
    tables = [np.array([0x1234], dtype='<i2')]
    table_len = 3
    expected_output = bytes([0x34, 0x12, 0x00, 0x00, 0x00, 0x00])
    result = compress._pack_lookup_tables(tables, table_len)
    self.assertEqual(result, expected_output)

  def test_float16_with_padding(self):
    tables = [np.array([1.5], dtype='<f2')]
    table_len = 3
    expected_output = bytes([0x00, 0x3e, 0x00, 0x00, 0x00, 0x00])
    result = compress._pack_lookup_tables(tables, table_len)
    self.assertEqual(result, expected_output)

  def test_multiple_tables_with_padding(self):
    tables = [np.array([0x1234], dtype='<i2'), np.array([0x5678], dtype='<i2')]
    table_len = 3
    expected_output = bytes([
        0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x78, 0x56, 0x00, 0x00, 0x00, 0x00
    ])
    result = compress._pack_lookup_tables(tables, table_len)
    self.assertEqual(result, expected_output)


# yapf: disable
TEST_MODEL = {
    "operator_codes": {
        0: {
            "builtin_code": tflite.BuiltinOperator.ADD,
        },
    },
    "metadata": {
        0: {
            "name": "metadata0",
            "buffer": 0
        },
    },
    "subgraphs": {
        0: {
            "operators": {
                0: {
                    "opcode_index": 0,
                    "inputs": (
                        0,
                        1,
                    ),
                    "outputs": (2, ),
                },
            },
            "tensors": {
                0: {
                    "shape": (16, 1),
                    "type": tflite.TensorType.UINT8,
                    "buffer": 1,
                    "quantization": {
                        "quantized_dimension": 1,
                        "scale": (1,),
                        "zero_point": (0,),
                    },
                },
                1: {
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT8,
                    "buffer": 2,
                    "quantization": {
                        "quantized_dimension": 1,
                        "scale": (1,),
                        "zero_point": (0,),
                    },
                },
                2: {
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT16,
                    "buffer": 3,
                    "quantization": {
                        "quantized_dimension": 1,
                        "scale": (1,),
                        "zero_point": (0,),
                    },
                },
                3: {
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT32,
                    "buffer": 4,
                    "quantization": {
                        "quantized_dimension": 1,
                        "scale": (1,),
                        "zero_point": (0,),
                    },
                },
                4: {
                    "shape": (16, 1),
                    "type": tflite.TensorType.INT32,
                    "buffer": 5,
                    "quantization": {
                        "quantized_dimension": 1,
                        "scale": (1,),
                        "zero_point": (0,),
                    },
                },
                5: {
                    "shape": (4, 5),
                    "type": tflite.TensorType.INT16,
                    "buffer": 6,
                    "quantization": {
                        "quantized_dimension": 1,
                        "scale": (1, 1, 1, 1, 1),
                        "zero_point": (0, 0, 0, 0, 0),
                    },
                },
                6: {
                    "shape": (5, 4),
                    "type": tflite.TensorType.INT16,
                    "buffer": 7,
                    "quantization": {
                        "quantized_dimension": 0,
                        "scale": (1, 1, 1, 1, 1),
                        "zero_point": (0, 0, 0, 0, 0),
                    },
                },
                7: {
                    "shape": (5, 4),
                    "type": tflite.TensorType.INT16,
                    "buffer": 8,
                    "quantization": {
                        "quantized_dimension": 0,
                        "scale": (1,),
                        "zero_point": (0,),
                    },
                },
                8: {
                    "shape": (16, 1),
                    "type": tflite.TensorType.UINT8,
                    "buffer": 9,
                },
            },
        },
    },
    "buffers": {
        0: None,

        1: np.array(range(16), dtype=np.dtype("<u1")),

        2: np.array(range(-16, 0), dtype=np.dtype("<i1")),

        3: np.array(range(-1616, -1600), dtype=np.dtype("<i2")),

        4: np.array(range(-160_016, -160_000), dtype=np.dtype("<i4")),

        5: np.array(range(16), dtype=np.dtype("<i4")),

        6: np.array(((1, 5, 9,  13, 17),
                     (2, 6, 10, 14, 18),
                     (3, 7, 11, 15, 19),
                     (4, 8, 12, 16, 20)), dtype=np.dtype("<i2")),

        7: np.array(((1,  2,  3,  4),
                     (5,  6,  7,  8),
                     (9,  10, 11, 12),
                     (13, 14, 15, 16),
                     (17, 18, 19, 20)), dtype=np.dtype("<i2")),

        8: np.array(((1, 2, 3, 4),
                     (1, 2, 3, 4),
                     (1, 2, 3, 4),
                     (1, 2, 3, 4),
                     (1, 2, 3, 4)), dtype=np.dtype("<i2")),

        9: np.array(range(16), dtype=np.dtype("<u1")),
    },
}

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
# yapf: enable


class TestsCompression(tf.test.TestCase):
  """Tests with the uncompressed model."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.flatbuffer = test_models.build(TEST_MODEL)
    cls.uncompressed = model_facade.read(cls.flatbuffer)

  def test_compression_metadata(self):
    """The compressed model has compression metadata."""
    compressed = compress.compress(self.flatbuffer, TEST_COMPRESSION_SPEC)
    model = model_facade.read(compressed)
    self.assertIn("metadata0", self.uncompressed.metadata)
    self.assertIn(compress.TFLITE_METADATA_KEY, model.metadata)

  def test_smaller_bitwidth(self):
    """Specifying LUT compression with too small a bitwidth fails"""
    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=1,
            compression=[spec.LookUpTableCompression(index_bitwidth=3)],
        ),
    ]
    self.assertRaises(compress.CompressionError,
                      lambda: compress.compress(self.flatbuffer, specs))

  def test_larger_bitwidth(self):
    """Specifying LUT compression with too large a bitwidth succeeds"""
    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=1,
            compression=[spec.LookUpTableCompression(index_bitwidth=5)],
        ),
    ]
    _ = compress.compress(self.flatbuffer, specs)

  def test_invalid_tensor_spec(self):
    """Specifying a tensor that doesn't exist raises CompressonError."""
    specs = [
        spec.Tensor(
            subgraph=666,
            tensor=1,
            compression=[spec.LookUpTableCompression(index_bitwidth=4)],
        ),
    ]
    self.assertRaises(compress.CompressionError,
                      lambda: compress.compress(self.flatbuffer, specs))

    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=666,
            compression=[spec.LookUpTableCompression(index_bitwidth=4)],
        ),
    ]
    self.assertRaises(compress.CompressionError,
                      lambda: compress.compress(self.flatbuffer, specs))

  def test_no_axis(self):
    """Raises if no quantization from which to infer compression axis."""
    specs = [
        spec.Tensor(
            subgraph=0,
            tensor=8,
            compression=[spec.LookUpTableCompression(index_bitwidth=4)],
        ),
    ]
    self.assertRaises(compress.CompressionError,
                      lambda: compress.compress(self.flatbuffer, specs))


class TestLutCompressedArray(tf.test.TestCase):

  def test_bitwidth(self):
    """Bitwidth is determined from index values."""
    a = compress._LutCompressedArray()
    a.indices = np.array((0, 1, 2, 3))
    self.assertEqual(a.index_bitwidth, 2)

    a.indices = np.array((0, 1, 2, 3, 4))
    self.assertEqual(a.index_bitwidth, 3)

    a.indices = np.array((0, 1, 1, 2, 2))
    self.assertEqual(a.index_bitwidth, 2)

    a.indices = np.array((0, 0, 0, 0))
    self.assertEqual(a.index_bitwidth, 1)


class TestCompressedModel(tf.test.TestCase):
  """Test the compressed model."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Create a model
    uncompressed_fb = test_models.build(TEST_MODEL)
    cls.uncompressed = model_facade.read(uncompressed_fb)

    # Compress the model
    compressed_fb = compress.compress(uncompressed_fb, TEST_COMPRESSION_SPEC)
    cls.compressed = model_facade.read(compressed_fb)

    # Extract the compression metadata
    metadata_flatbuffer = cls.compressed.metadata[compress.TFLITE_METADATA_KEY]
    cls.metadata = schema.MetadataT.InitFromPackedBuf(metadata_flatbuffer.data,
                                                      0)

  def test_uncompressed_tensors(self):
    """Tensors not in compression spec are not compressed.
    """
    # For all tensors in all subgraphs
    for subgraph in self.uncompressed.subgraphs:
      lut_tensors = self.metadata.subgraphs[subgraph.index].lutTensors

      for tensor in subgraph.tensors:
        # Search through specs
        match = lambda s: (s.subgraph == subgraph.index and s.tensor == tensor.
                           index)
        spec = next((s for s in TEST_COMPRESSION_SPEC if match(s)), None)

        # If the tensor is not in specs
        if spec is None:
          # Search through compression metadata
          match = lambda t: t.tensor == tensor.index
          metadata = next((t for t in lut_tensors if match(t)), None)

          # The tensor should not appear in compresion metadata
          self.assertIsNone(metadata)

  def _get_compressed(
      self, *, subgraph: int,
      tensor: int) -> tuple[int, bitarray.bitarray, np.ndarray]:
    """Helper: extracts the compressed tensor parts for a given spec.

    Returns:
      bitwidth
      indices
      values
    """
    subgraph_obj = self.compressed.subgraphs[subgraph]
    tensor_obj = subgraph_obj.tensors[tensor]
    lut_tensors = self.metadata.subgraphs[subgraph_obj.index].lutTensors
    lut_tensor = next(t for t in lut_tensors if t.tensor == tensor_obj.index)
    bitwidth = lut_tensor.indexBitwidth

    indices = bitarray.bitarray(buffer=tensor_obj.buffer.data, endian="big")
    n_indices = np.prod(tensor_obj.shape)
    indices = indices[:n_indices * bitwidth]  # trim possible padding

    value_buffer = self.compressed.buffers[lut_tensor.valueBuffer]
    values = np.frombuffer(value_buffer.data, dtype=tensor_obj.dtype)

    return bitwidth, indices, values

  def _make_indices(self, s: str) -> bitarray.bitarray:
    """Helper: makes indices from "01" strings for use as expected values."""
    return bitarray.bitarray(s, endian="big")

  def test_compressed_uint8(self):
    bitwidth, indices, values = self._get_compressed(subgraph=0, tensor=0)
    self.assertEqual(bitwidth, 4)

    # yapf: disable
    expected_indices = self._make_indices("""
      0000 0001 0010 0011
      0100 0101 0110 0111
      1000 1001 1010 1011
      1100 1101 1110 1111
    """)
    # yapf: enable
    self.assertEqual(indices, expected_indices)

    expected_values = np.array(range(16), dtype="<u1")
    self.assertAllEqual(values, expected_values)

  def test_compressed_int8(self):
    bitwidth, indices, values = self._get_compressed(subgraph=0, tensor=1)
    self.assertEqual(bitwidth, 4)

    # yapf: disable
    expected_indices = self._make_indices("""
      0000 0001 0010 0011
      0100 0101 0110 0111
      1000 1001 1010 1011
      1100 1101 1110 1111
    """)
    # yapf: enable
    self.assertEqual(indices, expected_indices)

    expected_values = np.array(range(-16, 0), dtype="<i1")
    self.assertAllEqual(values, expected_values)

  def test_compressed_int16(self):
    bitwidth, indices, values = self._get_compressed(subgraph=0, tensor=2)
    self.assertEqual(bitwidth, 4)

    # yapf: disable
    expected_indices = self._make_indices("""
      0000 0001 0010 0011
      0100 0101 0110 0111
      1000 1001 1010 1011
      1100 1101 1110 1111
    """)
    # yapf: enable
    self.assertEqual(indices, expected_indices)

    expected_values = np.array(range(-1616, -1600), dtype="<i2")
    self.assertAllEqual(values, expected_values)

  def test_compressed_int32(self):
    bitwidth, indices, values = self._get_compressed(subgraph=0, tensor=3)
    self.assertEqual(bitwidth, 4)

    # yapf: disable
    expected_indices = self._make_indices("""
      0000 0001 0010 0011
      0100 0101 0110 0111
      1000 1001 1010 1011
      1100 1101 1110 1111
    """)
    # yapf: enable
    self.assertEqual(indices, expected_indices)

    expected_values = np.array(range(-160_016, -160_000), dtype="<i4")
    self.assertAllEqual(values, expected_values)

  def test_axis_1(self):
    """Compression along quanitzation_dimension == 1."""
    bitwidth, indices, values = self._get_compressed(subgraph=0, tensor=5)
    self.assertEqual(bitwidth, 2)

    # yapf: disable
    expected_indices = self._make_indices("""
      00 00 00 00 00
      01 01 01 01 01
      10 10 10 10 10
      11 11 11 11 11
    """)
    # yapf: enable
    self.assertEqual(indices, expected_indices)

    expected_values = np.array(range(1, 21), dtype=np.dtype("<i2"))
    self.assertAllEqual(values, expected_values)

  def test_axis_0(self):
    """Compression along quanitzation_dimension == 0."""
    bitwidth, indices, values = self._get_compressed(subgraph=0, tensor=6)
    self.assertEqual(bitwidth, 2)

    # yapf: disable
    expected_indices = self._make_indices("""
      00 01 10 11
      00 01 10 11
      00 01 10 11
      00 01 10 11
      00 01 10 11
    """)
    # yapf: enable
    self.assertEqual(indices, expected_indices)

    expected_values = np.array(range(1, 21), dtype=np.dtype("<i2"))
    self.assertAllEqual(values, expected_values)

  def test_per_tensor(self):
    """Compression with one value table per tensor."""
    bitwidth, indices, values = self._get_compressed(subgraph=0, tensor=7)
    self.assertEqual(bitwidth, 2)

    # yapf: disable
    expected_indices = self._make_indices("""
      00 01 10 11
      00 01 10 11
      00 01 10 11
      00 01 10 11
      00 01 10 11
    """)
    # yapf: enable
    self.assertEqual(indices, expected_indices)

    expected_values = np.array(range(1, 5), dtype=np.dtype("<i2"))
    self.assertAllEqual(values, expected_values)


if __name__ == "__main__":
  tf.test.main()
