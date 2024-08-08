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
from tensorflow.lite.micro.compression import original_flatbuffer_py_generated as schema


def main():
  # The classes with a `T` suffix provide an object-oriented representation of
  # the object tree in the flatbuffer using native data structures.
  bq0_options = schema.BinQuantBufferOptionsT()
  bq0_options.valueTableIndex = 0
  bq0_options.compressedBitWidth = 2

  bq1_options = schema.BinQuantBufferOptionsT()
  bq1_options.valueTableIndex = 1
  bq1_options.compressedBitBidth = 4

  buffer0 = schema.CompressedBufferT()
  buffer0.bufferIndex = 0
  buffer0.options = bq0_options
  buffer0.optionsType = schema.CompressedBufferOptions.BinQuantBufferOptions

  buffer1 = schema.CompressedBufferT()
  buffer1.bufferIndex = 1
  buffer1.options = bq1_options
  buffer1.optionsType = schema.CompressedBufferOptions.BinQuantBufferOptions

  valuesInt8 = schema.ValuesInt8T()
  valuesInt8.values = [65]
  values0 = schema.ValuesT()
  values0.values = valuesInt8
  values0.values.Type = schema.ValuesUnion.ValuesInt8

  bq_compression = schema.BinQuantCompressionT()
  bq_compression.valueTables = [values0]

  metadata = schema.CompressionMetadataT()
  metadata.buffers = [buffer0, buffer1]
  metadata.binQuantCompression = bq_compression

  # Build the flatbuffer itself using the flatbuffers runtime module.
  builder = flatbuffers.Builder(32)
  root = metadata.Pack(builder)
  builder.Finish(root)
  buffer: bytearray = builder.Output()

  print(hexdump.hexdump(buffer, result='return'))
  print(f"length: {len(buffer)}")

  readback = schema.CompressionMetadataT.InitFromPackedBuf(buffer, 0)

  sys.exit()


if __name__ == "__main__":
  main()
