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

# Test validity of the flatbuffer schema and illustrate use of the flatbuffer
# machinery with Python

import sys
import hexdump
import flatbuffers

# `.*_generated` is the name of the module created by the Bazel rule
# `flatbuffer_py_library' based on the schema.
from tensorflow.lite.micro.compression import metadata_py_generated as schema


def main():
  # The classes with a `T` suffix provide an object-oriented representation of
  # the object tree in the flatbuffer using native data structures.
  lut_tensor0 = schema.LutTensorT()
  lut_tensor0.tensor = 63
  lut_tensor0.valueBuffer = 128
  lut_tensor0.indexBitwidth = 2

  lut_tensor1 = schema.LutTensorT()
  lut_tensor1.tensor = 64
  lut_tensor1.valueBuffer = 129
  lut_tensor1.indexBitwidth = 4

  subgraph0 = schema.SubgraphT()
  subgraph0.lutTensors = [lut_tensor0, lut_tensor1]

  metadata = schema.MetadataT()
  metadata.subgraphs = [subgraph0]

  # Build the flatbuffer itself using the flatbuffers runtime module.
  builder = flatbuffers.Builder(32)
  root = metadata.Pack(builder)
  builder.Finish(root)
  buffer: bytearray = builder.Output()

  print(hexdump.hexdump(buffer, result='return'))
  print(f"length: {len(buffer)}")

  read_metadata = schema.MetadataT.InitFromPackedBuf(buffer, 0)
  read_subgraph0 = read_metadata.subgraphs[0]

  def attrs_equal(a, b):
    return all(vars(a)[key] == vars(b)[key] for key in vars(a))

  assert attrs_equal(read_subgraph0.lutTensors[0], lut_tensor0)
  assert attrs_equal(read_subgraph0.lutTensors[1], lut_tensor1)

  sys.exit()


if __name__ == "__main__":
  main()
