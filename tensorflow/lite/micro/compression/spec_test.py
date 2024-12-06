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

import tensorflow as tf

from tflite_micro.tensorflow.lite.micro.compression import spec

# This corresponds to spec.EXAMPLE_YAML_SPEC
EXPECTED_PYTHON_SPEC = [
    spec.Tensor(subgraph=0,
                tensor=42,
                compression=[spec.LookUpTableCompression(index_bitwidth=4)]),
    spec.Tensor(subgraph=0,
                tensor=55,
                compression=[spec.LookUpTableCompression(index_bitwidth=2)]),
]


class TestLoadYaml(tf.test.TestCase):

  def testExampleSpec(self):
    result = spec.parse_yaml(spec.EXAMPLE_YAML_SPEC)
    self.assertEqual(result, EXPECTED_PYTHON_SPEC)

  def testMalformedYAML(self):
    bad = spec.EXAMPLE_YAML_SPEC + "  & foobar: 0"
    self.assertRaises(spec.ParseError, lambda: spec.parse_yaml(bad))

  def testUnexpectedType(self):
    bad = spec.EXAMPLE_YAML_SPEC + "  - subgraph: 'foobar'"
    self.assertRaises(spec.ParseError, lambda: spec.parse_yaml(bad))

  def testMissingFields(self):
    bad = spec.EXAMPLE_YAML_SPEC + "  - foobar: 0"
    self.assertRaises(spec.ParseError, lambda: spec.parse_yaml(bad))

  def testIgnoreExtraKeys(self):
    result = spec.parse_yaml(spec.EXAMPLE_YAML_SPEC + "foobar: 0")
    self.assertEqual(result, EXPECTED_PYTHON_SPEC)


if __name__ == "__main__":
  tf.test.main()
