/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "tensorflow/lite/schema/schema_generated.h"

const bool dump_buffers = false;

void dump_model_buffers(const tflite::Model* model) {
  fprintf(stderr, "model version: 0x%x\n", model->version());
  // This is a pointer to a vector of offsets:
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>* buffers =
      model->buffers();
  fprintf(stderr, "buffers: 0x%p\n", buffers);
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>&
      buffer_offsets = *buffers;
  int number_of_buffers = buffer_offsets.size();
  fprintf(stderr, "number of buffers: %d\n", buffer_offsets.size());
  for (int i = 0; i < number_of_buffers; ++i) {
    // C++ magic returns the actual buffer pointer here, rather than the
    // expected Offset that the Vector seems to hold:
    const tflite::Buffer* buffer = buffer_offsets[i];
    const flatbuffers::Vector<uint8_t>* data = buffer->data();
    // Only the weight buffers are allocated in the flatbuffer:
    if (data) {
      size_t buffer_size = data->size();
      const uint8_t* buffer_addr = data->Data();
      int buffer_offset = buffer_addr - reinterpret_cast<const uint8_t*>(model);
      fprintf(stderr, "buffer %d size: %zu, addr: 0x%p, offset: 0x%x\n", i,
              buffer_size, buffer_addr, buffer_offset);
      fprintf(stderr, "buffer contents: %x %x %x %x %x %x %x %x\n",
              buffer_addr[0], buffer_addr[1], buffer_addr[2], buffer_addr[3],
              buffer_addr[4], buffer_addr[5], buffer_addr[6], buffer_addr[7]);
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s input_flatbuffer output_flatbuffer\n", argv[0]);
    exit(-1);
  }
  std::string model_file;
  // Read the file into a string using the included util API call:
  flatbuffers::LoadFile(argv[1], false, &model_file);
  fprintf(stderr, "Original model size: %lu\n", model_file.size());
  // Parse the string into a C++ class.  Model is the root object of a tflite
  // flatbuffer file.
  const tflite::Model* model = tflite::GetModel(model_file.c_str());
  if (dump_buffers) dump_model_buffers(model);
  // A packed model is basically the file format mmaped into memory.
  // Unpacking it and then packing it with the C++ API should yield
  // a file with the force_align attributes respected.
  // ModelT is just the unpacked version of the model file.
  tflite::ModelT* unpacked_model = model->UnPack();
  flatbuffers::FlatBufferBuilder fbb;
  auto new_model = tflite::Model::Pack(fbb, unpacked_model);
  fbb.Finish(new_model, tflite::ModelIdentifier());
  const tflite::Model* aligned_model = tflite::GetModel(fbb.GetBufferPointer());
  if (dump_buffers) dump_model_buffers(aligned_model);
  flatbuffers::SaveFile(argv[2],
                        reinterpret_cast<char*>(fbb.GetBufferPointer()),
                        fbb.GetSize(), /*binary*/ true);
  int size = fbb.GetSize();
  fprintf(stderr, "Aligned model size: %d\n", size);
  return 0;
}
