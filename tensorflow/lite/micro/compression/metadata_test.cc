// Copyright 2024 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Test validity of the flatbuffer schema and illustrate use of the flatbuffer
// machinery with C++.

#include "tensorflow/lite/micro/compression/metadata_saved.h"
#include "tensorflow/lite/micro/hexdump.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/span.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

using tflite::micro::compression::LutTensor;
using tflite::micro::compression::LutTensorT;
using tflite::micro::compression::Metadata;
using tflite::micro::compression::MetadataT;
using tflite::micro::compression::Subgraph;
using tflite::micro::compression::SubgraphT;

namespace {

struct ExpectedLutTensor {
  int tensor;
  unsigned value_buffer;
  int index_bitwidth;
};

constexpr ExpectedLutTensor kExpected0 = {
    .tensor = 63,
    .value_buffer = 128,
    .index_bitwidth = 2,
};

constexpr ExpectedLutTensor kExpected1 = {
    .tensor = 64,
    .value_buffer = 129,
    .index_bitwidth = 4,
};

bool operator==(const ExpectedLutTensor& a, const LutTensor& b) {
  return a.tensor == b.tensor() && a.value_buffer == b.value_buffer() &&
         a.index_bitwidth == b.index_bitwidth();
}

constexpr unsigned kExpectedSchemaVersion = 1;

}  // end anonymous namespace

TF_LITE_MICRO_TESTS_BEGIN

// Create these objects on the stack and copy them into the subgraph's vector,
// so they can be compared later to what is read from the flatbuffer.
LutTensorT lut_tensor0;
lut_tensor0.tensor = kExpected0.tensor;
lut_tensor0.value_buffer = kExpected0.value_buffer;
lut_tensor0.index_bitwidth = kExpected0.index_bitwidth;

LutTensorT lut_tensor1;
lut_tensor1.tensor = kExpected1.tensor;
lut_tensor1.value_buffer = kExpected1.value_buffer;
lut_tensor1.index_bitwidth = kExpected1.index_bitwidth;

auto subgraph0 = std::make_unique<SubgraphT>();
subgraph0->lut_tensors.push_back(std::make_unique<LutTensorT>(lut_tensor0));
subgraph0->lut_tensors.push_back(std::make_unique<LutTensorT>(lut_tensor1));

auto metadata = std::make_unique<MetadataT>();
metadata->subgraphs.push_back(std::move(subgraph0));

flatbuffers::FlatBufferBuilder builder;
auto root = Metadata::Pack(builder, metadata.get());
builder.Finish(root);
auto flatbuffer = tflite::Span<const uint8_t>{
    reinterpret_cast<const uint8_t*>(builder.GetBufferPointer()),
    builder.GetSize()};

TF_LITE_MICRO_TEST(ReadbackEqualsWrite) {
  const Metadata* read_metadata =
      tflite::micro::compression::GetMetadata(flatbuffer.data());
  const Subgraph* read_subgraph0 = read_metadata->subgraphs()->Get(0);
  const LutTensor* read_lut_tensor0 = read_subgraph0->lut_tensors()->Get(0);
  const LutTensor* read_lut_tensor1 = read_subgraph0->lut_tensors()->Get(1);
  TF_LITE_MICRO_EXPECT(kExpected0 == *read_lut_tensor0);
  TF_LITE_MICRO_EXPECT(kExpected1 == *read_lut_tensor1);

  TF_LITE_MICRO_EXPECT(read_metadata->schema_version() ==
                       kExpectedSchemaVersion);

  // Print representation of the binary flatbuffer for debugging purposes.
  tflite::hexdump(flatbuffer);
  MicroPrintf("length: %i", flatbuffer.size());
}

TF_LITE_MICRO_TESTS_END
