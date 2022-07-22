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


int main(int argc, char** argv) {
  if (argc < 1) {
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
  // A packed model is basically the file format mmaped into memory.
  // Unpacking it and then packing it with the C++ API should yield
  // a file with the force_align attributes respected.
  // ModelT is just the unpacked version of the model file.
  tflite::ModelT* unpacked_model = model->UnPack();
  flatbuffers::FlatBufferBuilder fbb;
  auto new_model = tflite::Model::Pack(fbb, unpacked_model);
  fbb.Finish(new_model, tflite::ModelIdentifier());
  const tflite::Model* aligned_model = tflite::GetModel(fbb.GetBufferPointer());
  // flatbuffers::SaveFile(argv[2],
  //                       reinterpret_cast<char*>(fbb.GetBufferPointer()),
  //                       fbb.GetSize(), /*binary*/ true);
  int size = fbb.GetSize();
  fprintf(stderr, "New Optimized model size: %d\n", size);
  return 0;
}
