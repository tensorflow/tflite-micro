/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>

#include "codegen/preprocessor/preprocessor_schema_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

std::unique_ptr<char[]> ReadModelFile(const char* model_file_name) {
  std::ifstream model_file(model_file_name, std::ios::binary);
  if (!model_file.is_open()) {
    std::cerr << "codegen_preprocessor: could not open model file: "
              << model_file_name << std::endl;
    return nullptr;
  }

  model_file.seekg(0, std::ios::end);
  size_t num_bytes = model_file.tellg();
  model_file.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> model_data(new char[num_bytes]);
  model_file.read(model_data.get(), num_bytes);

  return model_data;
}

int WriteOutputFile(const char* output_file_name,
                    flatbuffers::span<uint8_t> output) {
  std::ofstream output_file(output_file_name, std::ios::trunc);
  if (!output_file.is_open()) {
    std::cerr << "codegen_preprocessor: could not open output file: "
              << output_file_name << std::endl;
    return EXIT_FAILURE;
  }

  output_file.write(reinterpret_cast<char*>(output.data()), output.size());
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "codegen_preprocessor: invalid usage!" << std::endl;
    std::cerr << "usage: codegen_preprocessor <tflite_model> <output_file>"
              << std::endl;
    return EXIT_FAILURE;
  }

  const char* model_file_name = argv[1];
  const char* output_file_name = argv[2];

  const auto model_data = ReadModelFile(model_file_name);
  if (!model_data) {
    return EXIT_FAILURE;
  }

  // We have to create our own allocator, as the typical TFLM runtime disables
  // its use (to avoid dynamic allocation).
  flatbuffers::DefaultAllocator allocator;
  flatbuffers::FlatBufferBuilder builder{2048, &allocator};
  const auto input_model_path = builder.CreateString(model_file_name);

  // Do the preprocess work.

  tflm::codegen::preprocessor::DataBuilder data_builder(builder);
  data_builder.add_input_model_path(input_model_path);
  builder.Finish(data_builder.Finish());

  return WriteOutputFile(output_file_name, builder.GetBufferSpan());
}
