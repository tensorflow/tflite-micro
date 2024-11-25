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

import numpy as np
import tensorflow as tf

from tflite_micro.tensorflow.lite.micro.compression import compress


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


if __name__ == "__main__":
  tf.test.main()
