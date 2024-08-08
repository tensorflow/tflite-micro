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

using tflite::micro::compression::Metadata;
using tflite::micro::compression::MetadataT;
using tflite::micro::compression::Subgraph;
using tflite::micro::compression::SubgraphT;
using tflite::micro::compression::LutTensor;
using tflite::micro::compression::LutTensorT;

bool operator==(const LutTensorT& a, const LutTensor& b) {
  return 
    a.tensor == b.tensor() &&
    a.value_buffer == b.value_buffer() &&
    a.index_bitwidth == b.index_bitwidth();
}

int main(int argc, char* argv[]) {
  // Create these objects on the stack and copy them into the subgraph's vector
  // later, so that we can compare to these objects to what we read from the
  // flatbuffer later.
  LutTensorT lut_tensor0;
  lut_tensor0.tensor = 63;
  lut_tensor0.value_buffer = 128;
  lut_tensor0.index_bitwidth = 2;

  LutTensorT lut_tensor1;
  lut_tensor1.tensor = 64;
  lut_tensor1.value_buffer = 129;
  lut_tensor1.index_bitwidth = 4;

  auto subgraph0 = std::make_unique<SubgraphT>();
  subgraph0->lut_tensors.push_back(std::make_unique<LutTensorT>(lut_tensor0));
  subgraph0->lut_tensors.push_back(std::make_unique<LutTensorT>(lut_tensor1));

  auto metadata = std::make_unique<MetadataT>();
  metadata->subgraphs.push_back(std::move(subgraph0));

  flatbuffers::FlatBufferBuilder builder;
  auto root = Metadata::Pack(builder, metadata.get());
  builder.Finish(root);
  const uint8_t* buffer = builder.GetBufferPointer();
  const size_t buffer_size = builder.GetSize();

  tflite::hexdump(
      {reinterpret_cast<const std::byte*>(buffer), buffer_size});
  std::cout << "length: " << buffer_size << "\n";

  const Metadata* read_metadata = tflite::micro::compression::GetMetadata(buffer);
  const Subgraph* read_subgraph0 = read_metadata->subgraphs()->Get(0);
  const LutTensor* read_lut_tensor0 = read_subgraph0->lut_tensors()->Get(0);
  const LutTensor* read_lut_tensor1 = read_subgraph0->lut_tensors()->Get(1);
  assert(lut_tensor0 == *read_lut_tensor0);
  assert(lut_tensor1 == *read_lut_tensor1);

  return 0;
}
