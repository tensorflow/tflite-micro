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
#
"""Tests for the compression spec builder."""

import tensorflow as tf

from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.micro.compression import spec_builder


class SpecBuilderTest(tf.test.TestCase):

  def test_basic_builder_pattern(self):
    """Test basic fluent builder usage."""
    result = (spec_builder.SpecBuilder().add_tensor(
        subgraph=0, tensor=2).with_lut(index_bitwidth=4).add_tensor(
            subgraph=0, tensor=4).with_lut(index_bitwidth=2).build())

    self.assertEqual(len(result), 2)

    # Check first tensor
    self.assertEqual(result[0].subgraph, 0)
    self.assertEqual(result[0].tensor, 2)
    self.assertEqual(len(result[0].compression), 1)
    self.assertIsInstance(result[0].compression[0],
                          spec.LookUpTableCompression)
    self.assertEqual(result[0].compression[0].index_bitwidth, 4)

    # Check second tensor
    self.assertEqual(result[1].subgraph, 0)
    self.assertEqual(result[1].tensor, 4)
    self.assertEqual(len(result[1].compression), 1)
    self.assertIsInstance(result[1].compression[0],
                          spec.LookUpTableCompression)
    self.assertEqual(result[1].compression[0].index_bitwidth, 2)

  def test_non_chained_usage(self):
    """Test using builder without method chaining."""
    builder = spec_builder.SpecBuilder()
    builder.add_tensor(0, 2).with_lut(4)
    builder.add_tensor(0, 4).with_lut(2)
    result = builder.build()

    self.assertEqual(len(result), 2)
    self.assertEqual(result[0].tensor, 2)
    self.assertEqual(result[0].compression[0].index_bitwidth, 4)
    self.assertEqual(result[1].tensor, 4)
    self.assertEqual(result[1].compression[0].index_bitwidth, 2)

  def test_empty_spec(self):
    """Test building an empty spec."""
    result = spec_builder.SpecBuilder().build()
    self.assertEqual(len(result), 0)

  def test_single_tensor(self):
    """Test building a spec with just one tensor."""
    result = (spec_builder.SpecBuilder().add_tensor(
        subgraph=2, tensor=42).with_lut(index_bitwidth=16).build())

    self.assertEqual(len(result), 1)
    self.assertEqual(result[0].subgraph, 2)
    self.assertEqual(result[0].tensor, 42)
    self.assertEqual(result[0].compression[0].index_bitwidth, 16)

  def test_tensor_without_compression(self):
    """Test that tensors can be added without compression methods."""
    builder = spec_builder.SpecBuilder()
    # Add tensor but don't call with_lut
    builder.add_tensor(0, 1)
    builder.add_tensor(0, 2).with_lut(4)
    result = builder.build()

    self.assertEqual(len(result), 2)
    self.assertEqual(result[0].tensor, 1)
    self.assertEqual(len(result[0].compression), 0)
    self.assertEqual(result[1].tensor, 2)
    self.assertEqual(len(result[1].compression), 1)

  def test_builder_produces_same_type_as_parse_yaml(self):
    """Test that builder produces same data structure as parse_yaml."""
    # Build using the builder
    built_spec = (spec_builder.SpecBuilder().add_tensor(
        subgraph=0, tensor=42).with_lut(index_bitwidth=4).add_tensor(
            subgraph=0, tensor=55).with_lut(index_bitwidth=2).build())

    # Parse the example YAML from spec.py
    parsed_spec = spec.parse_yaml(spec.EXAMPLE_YAML_SPEC)

    # They should be equivalent
    self.assertEqual(len(built_spec), len(parsed_spec))
    for built, parsed in zip(built_spec, parsed_spec):
      self.assertEqual(built.subgraph, parsed.subgraph)
      self.assertEqual(built.tensor, parsed.tensor)
      self.assertEqual(len(built.compression), len(parsed.compression))
      self.assertEqual(built.compression[0].index_bitwidth,
                       parsed.compression[0].index_bitwidth)


if __name__ == "__main__":
  tf.test.main()
