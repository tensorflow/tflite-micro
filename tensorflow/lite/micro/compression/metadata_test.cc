/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Test validity of the flatbuffer schema and illustrate use of the flatbuffer
// machinery with C++.

#include <iostream>
#include <vector>

#include "metadata_generated.h"
#include "tensorflow/lite/micro/hexdump.h"

using tflite::micro::compression::LutTensor;
using tflite::micro::compression::Metadata;
using tflite::micro::compression::MetadataT;

bool operator==(const LutTensor& a, const LutTensor& b) {
  return 
    a.subgraph() == b.subgraph() &&
    a.tensor() == b.tensor() &&
    a.index_bitwidth() == b.index_bitwidth() &&
    a.index_buffer() == b.index_buffer() &&
    a.value_buffer() == b.value_buffer();
}

int main(int argc, char* argv[]) {
  const LutTensor lut_tensor0 {
    0,   // subgraph
    127, // tensor
    2,   // index_bitwidth
    128, // index_buffer
    129, // value_buffer
  };
  const LutTensor lut_tensor1 {
    1,   // subgraph
    164, // tensor
    2,   // index_bitwidth
    136, // index_buffer
    129, // value_buffer
  };
  MetadataT metadata;
  metadata.lut_tensors = {lut_tensor0, lut_tensor1};

  flatbuffers::FlatBufferBuilder builder;
  auto root = Metadata::Pack(builder, &metadata);
  builder.Finish(root);
  const uint8_t* buffer = builder.GetBufferPointer();

  tflite::hexdump(
      {reinterpret_cast<const std::byte*>(buffer), builder.GetSize()});
  std::cout << "length: " << builder.GetSize() << "\n";

  auto readback = tflite::micro::compression::GetMetadata(buffer);
  auto& read_lut_tensor0 = *readback->lut_tensors()->Get(0);
  auto& read_lut_tensor1 = *readback->lut_tensors()->Get(1);
  assert(read_lut_tensor0 == lut_tensor0);
  assert(read_lut_tensor1 == lut_tensor1);

  return 0;
}
