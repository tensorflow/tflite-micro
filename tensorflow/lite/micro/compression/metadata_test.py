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
"""
Test validity of the flatbuffer schema and illustrate the use of the
flatbuffer machinery with Python.
"""

from dataclasses import dataclass
import flatbuffers
import tensorflow as tf

# `.*_generated` is the name of the module created by the Bazel rule
# `flatbuffer_py_library' based on the schema.
from tflite_micro.tensorflow.lite.micro.compression import metadata_py_generated as schema


@dataclass
class _LutTensor:
  tensor: int
  valueBuffer: int
  indexBitwidth: int


_EXPECTED_0 = _LutTensor(
    tensor=63,
    valueBuffer=128,
    indexBitwidth=2,
)

_EXPECTED_1 = _LutTensor(
    tensor=64,
    valueBuffer=129,
    indexBitwidth=4,
)

# This is set in the schema definition.
_EXPECTED_SCHEMA_VERSION = 1


class TestReadEqualsWrite(tf.test.TestCase):

  def setUp(self):
    """Sets up the test by creating a flatbuffer using the metadata schema.
    """
    # The classes with a `T` suffix provide an object-oriented representation of
    # the object tree in the flatbuffer using native data structures.
    lut_tensor0 = schema.LutTensorT()
    lut_tensor0.tensor = _EXPECTED_0.tensor
    lut_tensor0.valueBuffer = _EXPECTED_0.valueBuffer
    lut_tensor0.indexBitwidth = _EXPECTED_0.indexBitwidth

    lut_tensor1 = schema.LutTensorT()
    lut_tensor1.tensor = _EXPECTED_1.tensor
    lut_tensor1.valueBuffer = _EXPECTED_1.valueBuffer
    lut_tensor1.indexBitwidth = _EXPECTED_1.indexBitwidth

    subgraph0 = schema.SubgraphT()
    subgraph0.lutTensors = [lut_tensor0, lut_tensor1]

    metadata = schema.MetadataT()
    metadata.subgraphs = [subgraph0]

    # Write the flatbuffer itself using the flatbuffers runtime module.
    builder = flatbuffers.Builder(32)
    root = metadata.Pack(builder)
    builder.Finish(root)
    self.flatbuffer: bytearray = builder.Output()

  def testLutTensors(self):
    """Reads back the LutTensors and ensures they match expected values.
    """
    # Read the flatbuffer using the flatbuffers runtime module.
    metadata = schema.MetadataT.InitFromPackedBuf(self.flatbuffer, 0)

    read_tensor0 = metadata.subgraphs[0].lutTensors[0]
    self.assertEqual(read_tensor0.tensor, _EXPECTED_0.tensor)
    self.assertEqual(read_tensor0.valueBuffer, _EXPECTED_0.valueBuffer)
    self.assertEqual(read_tensor0.indexBitwidth, _EXPECTED_0.indexBitwidth)

    read_tensor1 = metadata.subgraphs[0].lutTensors[1]
    self.assertEqual(read_tensor1.tensor, _EXPECTED_1.tensor)
    self.assertEqual(read_tensor1.valueBuffer, _EXPECTED_1.valueBuffer)
    self.assertEqual(read_tensor1.indexBitwidth, _EXPECTED_1.indexBitwidth)

  def testSchemaVersion(self):
    """Reads back the LutTensors and ensures they match expected values.
    """
    # Read the flatbuffer using the flatbuffers runtime module.
    metadata = schema.MetadataT.InitFromPackedBuf(self.flatbuffer, 0)

    self.assertEqual(metadata.schemaVersion, _EXPECTED_SCHEMA_VERSION)

  def testPrintFlatbufferLen(self):
    """Print the flatbuffer length for the log.
    """
    print(f"length: {len(self.flatbuffer)}")


if __name__ == "__main__":
  tf.test.main()
