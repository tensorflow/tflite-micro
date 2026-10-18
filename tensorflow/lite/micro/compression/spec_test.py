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

import unittest

from tflite_micro.tensorflow.lite.micro.compression import spec

# This corresponds to spec.EXAMPLE_YAML_SPEC
EXPECTED_PYTHON_SPEC = [
  spec.Tensor(
    subgraph=0,
    tensor=42,
    compression=[
      spec.LookUpTableCompression(
        index_bitwidth=4, mode=spec.PerChannel(axis=0)
      )
    ],
  ),
  spec.Tensor(
    subgraph=0,
    tensor=55,
    compression=[
      spec.LookUpTableCompression(index_bitwidth=2, mode=spec.PerTensor())
    ],
  ),
]


def _lut_spec(*lut_lines: str) -> str:
  """Returns a one-tensor spec whose lut entry holds the given lines."""
  lines = [
    "tensors:",
    "  - subgraph: 0",
    "    tensor: 0",
    "    compression:",
    "      - lut:",
  ]
  lines += [" " * 10 + line for line in lut_lines]
  return "\n".join(lines) + "\n"


class TestLutMode(unittest.TestCase):
  """Tests for parsing the per_tensor/per_channel choice."""

  def testMissingModeRaises(self):
    bad = _lut_spec("index_bitwidth: 4")
    with self.assertRaisesRegex(spec.ParseError, "per_tensor or per_channel"):
      spec.parse_yaml(bad)

  def testBothModesRaise(self):
    bad = _lut_spec(
      "index_bitwidth: 4",
      "per_tensor:",
      "per_channel:",
      "  axis: 0",
    )
    with self.assertRaisesRegex(spec.ParseError, "contradictory"):
      spec.parse_yaml(bad)

  def testPerTensorWithPayloadRaises(self):
    bad = _lut_spec(
      "index_bitwidth: 4",
      "per_tensor: 1",
    )
    with self.assertRaisesRegex(spec.ParseError, "no value"):
      spec.parse_yaml(bad)

  def testPerChannelWithoutAxisRaises(self):
    bad = _lut_spec(
      "index_bitwidth: 4",
      "per_channel:",
    )
    with self.assertRaisesRegex(spec.ParseError, "axis"):
      spec.parse_yaml(bad)

  def testNegativeAxisRaises(self):
    bad = _lut_spec(
      "index_bitwidth: 4",
      "per_channel:",
      "  axis: -1",
    )
    with self.assertRaisesRegex(spec.ParseError, "non-negative"):
      spec.parse_yaml(bad)

  def testNonIntegerAxisRaises(self):
    bad = _lut_spec(
      "index_bitwidth: 4",
      "per_channel:",
      "  axis: zero",
    )
    with self.assertRaisesRegex(spec.ParseError, "non-negative"):
      spec.parse_yaml(bad)


if __name__ == "__main__":
  unittest.main()
