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

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "flatbuffers/util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace py = pybind11;

void align_tflite_model(const char* input_file_name,
                        const char* output_file_name) {
  std::string model_file;
  // Read the file into a string using the included util API call:
  flatbuffers::LoadFile(input_file_name, false, &model_file);
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
  flatbuffers::SaveFile(output_file_name,
                        reinterpret_cast<char*>(fbb.GetBufferPointer()),
                        fbb.GetSize(), /*binary*/ true);
}

PYBIND11_MODULE(tflite_flatbuffer_align_wrapper, m) {
  m.doc() = "tflite_flatbuffer_align_wrapper";
  m.def("align_tflite_model", &align_tflite_model,
        "Aligns the tflite flatbuffer to (16), by unpacking and repacking via "
        "the flatbuffer C++ API.",
        py::arg("input_file_name"), py::arg("output_file_name"));
}
