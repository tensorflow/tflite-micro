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
"""Unit tests for LUT compression plugin."""

import numpy as np
import tensorflow as tf

from tflite_micro.tensorflow.lite.micro.compression import compressor
from tflite_micro.tensorflow.lite.micro.compression import decode
from tflite_micro.tensorflow.lite.micro.compression import lut
from tflite_micro.tensorflow.lite.micro.compression import model_editor
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.python import schema_py_generated as tflite


class TestCompressArray(tf.test.TestCase):
  """Tests for the compress_array function."""

  def test_per_tensor_basic(self):
    """Per-tensor compression extracts unique values."""
    array = np.array([1, 2, 1, 2, 3, 3], dtype=np.int8)
    compressed = lut.compress_array(array, axis=None)

    self.assertIsNone(compressed.compression_axis)
    self.assertEqual(len(compressed.lookup_tables), 1)
    self.assertAllEqual(compressed.lookup_tables[0], [1, 2, 3])
    # Indices should map back to original values
    reconstructed = compressed.lookup_tables[0][compressed.indices]
    self.assertAllEqual(reconstructed, array)

  def test_per_tensor_preserves_shape(self):
    """Indices array has same shape as input."""
    # yapf: disable
    array = np.array([[1, 2],
                      [3, 1],
                      [2, 3]], dtype=np.int8)
    # yapf: enable
    compressed = lut.compress_array(array, axis=None)

    self.assertEqual(compressed.indices.shape, array.shape)

  def test_per_channel_axis0(self):
    """Per-channel compression along axis 0."""
    # Each row gets its own value table
    # yapf: disable
    array = np.array([[1, 1, 1],
                      [5, 5, 5],
                      [9, 9, 9]], dtype=np.int8)
    # yapf: enable
    compressed = lut.compress_array(array, axis=0)

    self.assertEqual(compressed.compression_axis, 0)
    self.assertEqual(len(compressed.lookup_tables), 3)
    self.assertAllEqual(compressed.lookup_tables[0], [1])
    self.assertAllEqual(compressed.lookup_tables[1], [5])
    self.assertAllEqual(compressed.lookup_tables[2], [9])

  def test_per_channel_axis1(self):
    """Per-channel compression along axis 1."""
    # Each column gets its own value table
    # yapf: disable
    array = np.array([[1, 5],
                      [1, 5],
                      [1, 5]], dtype=np.int8)
    # yapf: enable
    compressed = lut.compress_array(array, axis=1)

    self.assertEqual(compressed.compression_axis, 1)
    self.assertEqual(len(compressed.lookup_tables), 2)
    self.assertAllEqual(compressed.lookup_tables[0], [1])
    self.assertAllEqual(compressed.lookup_tables[1], [5])

  def test_single_value(self):
    """Array with single unique value."""
    array = np.array([7, 7, 7, 7], dtype=np.int8)
    compressed = lut.compress_array(array, axis=None)

    self.assertEqual(len(compressed.lookup_tables), 1)
    self.assertAllEqual(compressed.lookup_tables[0], [7])
    self.assertAllEqual(compressed.indices, [0, 0, 0, 0])

  def test_bitwidth_calculation(self):
    """Index bitwidth is computed correctly."""
    # 3 unique values -> 2 bits needed
    array = np.array([0, 1, 2], dtype=np.int8)
    compressed = lut.compress_array(array, axis=None)
    self.assertEqual(compressed.index_bitwidth, 2)

    # 4 unique values -> 2 bits needed
    array = np.array([0, 1, 2, 3], dtype=np.int8)
    compressed = lut.compress_array(array, axis=None)
    self.assertEqual(compressed.index_bitwidth, 2)

    # 5 unique values -> 3 bits needed
    array = np.array([0, 1, 2, 3, 4], dtype=np.int8)
    compressed = lut.compress_array(array, axis=None)
    self.assertEqual(compressed.index_bitwidth, 3)

  def test_bitwidth_single_value(self):
    """Single unique value requires 1 bit."""
    array = np.array([42, 42, 42], dtype=np.int8)
    compressed = lut.compress_array(array, axis=None)
    self.assertEqual(compressed.index_bitwidth, 1)


class TestPackIndices(tf.test.TestCase):
  """Tests for the pack_indices function."""

  def test_4bit_packing(self):
    """Pack indices into 4-bit fields."""
    indices = np.array([1, 2, 3, 0])
    result = lut.pack_indices(indices, bitwidth=4)
    # Big-endian: 0001 0010 | 0011 0000 = 0x12 0x30
    self.assertEqual(result, bytes([0x12, 0x30]))

  def test_2bit_packing(self):
    """Pack indices into 2-bit fields."""
    indices = np.array([0, 1, 2, 3])
    result = lut.pack_indices(indices, bitwidth=2)
    # Big-endian: 00 01 10 11 = 0x1B
    self.assertEqual(result, bytes([0x1B]))

  def test_3bit_packing(self):
    """Pack indices into 3-bit fields."""
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    result = lut.pack_indices(indices, bitwidth=3)
    # 000 001 010 011 | 100 101 110 111
    # 00000101 | 00111001 | 01110111 = 0x05 0x39 0x77
    self.assertEqual(result, bytes([0x05, 0x39, 0x77]))

  def test_1bit_packing(self):
    """Pack indices into 1-bit fields."""
    indices = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    result = lut.pack_indices(indices, bitwidth=1)
    # 0 1 0 1 1 0 1 0 = 0x5A
    self.assertEqual(result, bytes([0x5A]))

  def test_multidimensional_flattens(self):
    """Multidimensional indices are flattened row-major."""
    # yapf: disable
    indices = np.array([[0, 1],
                        [2, 3]])
    # yapf: enable
    result = lut.pack_indices(indices, bitwidth=4)
    # 0000 0001 | 0010 0011 = 0x01 0x23
    self.assertEqual(result, bytes([0x01, 0x23]))


class TestPackLookupTables(tf.test.TestCase):
  """Tests for the pack_lookup_tables function."""

  def test_single_table_int8(self):
    """Pack single INT8 lookup table."""
    tables = [np.array([10, 20, 30], dtype=np.int8)]
    result = lut.pack_lookup_tables(tables, table_len=4)
    # Values: 10, 20, 30, 0 (padding)
    self.assertEqual(result, bytes([10, 20, 30, 0]))

  def test_multiple_tables(self):
    """Pack multiple lookup tables."""
    tables = [
        np.array([1, 2], dtype=np.int8),
        np.array([3, 4], dtype=np.int8),
    ]
    result = lut.pack_lookup_tables(tables, table_len=4)
    # Table 1: 1, 2, 0, 0 | Table 2: 3, 4, 0, 0
    self.assertEqual(result, bytes([1, 2, 0, 0, 3, 4, 0, 0]))

  def test_int16_little_endian(self):
    """INT16 values are packed in native byte order."""
    tables = [np.array([0x1234, 0x5678], dtype='<i2')]
    result = lut.pack_lookup_tables(tables, table_len=2)
    # Little-endian: 0x34 0x12 0x78 0x56
    self.assertEqual(result, bytes([0x34, 0x12, 0x78, 0x56]))

  def test_no_padding_needed(self):
    """Table exactly fills the stride."""
    tables = [np.array([1, 2, 3, 4], dtype=np.int8)]
    result = lut.pack_lookup_tables(tables, table_len=4)
    self.assertEqual(result, bytes([1, 2, 3, 4]))


class TestIdentifyCompressionAxis(tf.test.TestCase):
  """Tests for identify_compression_axis function."""

  def test_per_tensor_quantization(self):
    """Single scale means per-tensor compression."""
    tensor = model_editor.Tensor(
        shape=(4, 4),
        dtype=tflite.TensorType.INT8,
        quantization=model_editor.Quantization(scales=0.5, zero_points=0),
    )
    axis = lut.identify_compression_axis(tensor)
    self.assertIsNone(axis)

  def test_per_channel_axis0(self):
    """Multiple scales on axis 0."""
    tensor = model_editor.Tensor(
        shape=(4, 8),
        dtype=tflite.TensorType.INT8,
        quantization=model_editor.Quantization(
            scales=[0.1, 0.2, 0.3, 0.4],
            zero_points=[0, 0, 0, 0],
            axis=0,
        ),
    )
    axis = lut.identify_compression_axis(tensor)
    self.assertEqual(axis, 0)

  def test_per_channel_axis1(self):
    """Multiple scales on axis 1."""
    tensor = model_editor.Tensor(
        shape=(4, 8),
        dtype=tflite.TensorType.INT8,
        quantization=model_editor.Quantization(
            scales=[0.1] * 8,
            zero_points=[0] * 8,
            axis=1,
        ),
    )
    axis = lut.identify_compression_axis(tensor)
    self.assertEqual(axis, 1)

  def test_no_quantization_raises(self):
    """Missing quantization raises CompressionError."""
    tensor = model_editor.Tensor(
        shape=(4, 4),
        dtype=tflite.TensorType.INT8,
    )
    with self.assertRaises(compressor.CompressionError):
      lut.identify_compression_axis(tensor)


class TestLutCompressor(tf.test.TestCase):
  """Tests for the LutCompressor class."""

  def test_decode_type(self):
    """LutCompressor returns DecodeType.LUT."""
    compressor_instance = lut.LutCompressor()
    self.assertEqual(compressor_instance.decode_type, decode.DecodeType.LUT)

  def test_compress_basic(self):
    """Basic compression produces valid result."""
    tensor = model_editor.Tensor(
        shape=(4, ),
        dtype=tflite.TensorType.INT8,
        data=np.array([1, 2, 1, 2], dtype=np.int8),
        quantization=model_editor.Quantization(scales=1.0, zero_points=0),
    )
    method = spec.LookUpTableCompression(index_bitwidth=4)

    compressor_instance = lut.LutCompressor()
    result = compressor_instance.compress(tensor, method)

    # Verify we got encoded data and ancillary data
    self.assertIsInstance(result.encoded_data, bytes)
    self.assertIsInstance(result.ancillary_data, bytes)
    self.assertGreater(len(result.encoded_data), 0)
    # Ancillary data should be at least 16 bytes (DCM header)
    self.assertGreaterEqual(len(result.ancillary_data), 16)

  def test_compress_ancillary_data_format(self):
    """Ancillary data matches C++ expected format."""
    tensor = model_editor.Tensor(
        shape=(4, ),
        dtype=tflite.TensorType.INT8,
        data=np.array([1, 2, 3, 4], dtype=np.int8),
        quantization=model_editor.Quantization(scales=1.0, zero_points=0),
    )
    method = spec.LookUpTableCompression(index_bitwidth=4)

    compressor_instance = lut.LutCompressor()
    result = compressor_instance.compress(tensor, method)

    # Parse DCM header
    dcm_bytes = result.ancillary_data[:16]
    self.assertEqual(dcm_bytes[0], 0)  # decode_type = LUT
    self.assertEqual(dcm_bytes[1], 1)  # DCM version
    self.assertEqual(dcm_bytes[4], 1)  # LUT version
    self.assertEqual(dcm_bytes[5] & 0x07, 4)  # bitwidth = 4
    self.assertEqual(dcm_bytes[6], 16)  # value_table_stride = 2^4

  def test_compress_bitwidth_too_small_raises(self):
    """Specifying too small bitwidth raises error."""
    # 16 unique values need 4 bits, but we specify 3
    tensor = model_editor.Tensor(
        shape=(16, ),
        dtype=tflite.TensorType.INT8,
        data=np.array(range(16), dtype=np.int8),
        quantization=model_editor.Quantization(scales=1.0, zero_points=0),
    )
    method = spec.LookUpTableCompression(index_bitwidth=3)

    compressor_instance = lut.LutCompressor()
    with self.assertRaises(compressor.CompressionError):
      compressor_instance.compress(tensor, method)

  def test_compress_wrong_method_type_raises(self):
    """Passing wrong compression method type raises error."""
    tensor = model_editor.Tensor(
        shape=(4, ),
        dtype=tflite.TensorType.INT8,
        data=np.array([1, 2, 1, 2], dtype=np.int8),
        quantization=model_editor.Quantization(scales=1.0, zero_points=0),
    )
    # Use base CompressionMethod instead of LookUpTableCompression
    method = spec.CompressionMethod()

    compressor_instance = lut.LutCompressor()
    with self.assertRaises(compressor.CompressionError):
      compressor_instance.compress(tensor, method)

  def test_compress_no_data_raises(self):
    """Tensor without data raises error."""
    tensor = model_editor.Tensor(
        shape=(4, ),
        dtype=tflite.TensorType.INT8,
        quantization=model_editor.Quantization(scales=1.0, zero_points=0),
    )
    method = spec.LookUpTableCompression(index_bitwidth=4)

    compressor_instance = lut.LutCompressor()
    with self.assertRaises(compressor.CompressionError):
      compressor_instance.compress(tensor, method)


class TestLutAncillaryData(tf.test.TestCase):
  """Tests for LutAncillaryData."""

  def test_to_user_data_format(self):
    """User data bytes match expected format."""
    lut_data = lut.LutAncillaryData(
        lut_version=1,
        bitwidth=4,
        value_table_stride=16,
        value_tables=b'',
    )
    user_data = lut_data.to_user_data()

    self.assertEqual(len(user_data), 12)
    self.assertEqual(user_data[0], 1)  # lut_version
    self.assertEqual(user_data[1], 4)  # bitwidth
    self.assertEqual(user_data[2], 16)  # stride

  def test_bitwidth_validation(self):
    """Bitwidth must be 1-7."""
    with self.assertRaises(ValueError):
      lut.LutAncillaryData(bitwidth=0)
    with self.assertRaises(ValueError):
      lut.LutAncillaryData(bitwidth=8)

  def test_stride_validation(self):
    """Stride must be 0-128."""
    with self.assertRaises(ValueError):
      lut.LutAncillaryData(value_table_stride=129)


if __name__ == "__main__":
  tf.test.main()
