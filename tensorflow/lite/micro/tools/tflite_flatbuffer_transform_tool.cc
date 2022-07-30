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

#include <iostream>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "tensorflow/lite/schema/schema_generated.h"

void StripStrings(tflite::ModelT* model) {
  /*Strips all nonessential strings from the model to reduce model size.

  We remove the following strings:
  (find strings by searching ":string" in the tensorflow lite flatbuffer schema)
  1. Model description
  2. SubGraph name
  3. Tensor names
  We retain OperatorCode custom_code and Metadata name.

  Args:
    model: The model from which to remove nonessential strings.
  */
  model->description.clear();
  model->signature_defs.clear();
  for (int subgraph_index = 0; subgraph_index < model->subgraphs.size();
       subgraph_index++) {
    model->subgraphs[subgraph_index]->name.clear();
    for (int tensor_index = 0;
         tensor_index < model->subgraphs[subgraph_index]->tensors.size();
         tensor_index++) {
      model->subgraphs[subgraph_index]->tensors[tensor_index]->name.clear();
    }
  }
}

void RemoveExtraneousQuantizationData(tflite::ModelT* model) {
  /*
  We remove the following arrays from weight tensors when quanzation paramaters
  aren't needed to reduce model size:
  1. max
  2. min
  3. zero_point

  Args:
    model: The model from which to remove nonessential quanzation data.
  */

  for (int subgraph_index = 0; subgraph_index < model->subgraphs.size();
       subgraph_index++) {
    for (int tensor_index = 0;
         tensor_index < model->subgraphs[subgraph_index]->tensors.size();
         tensor_index++) {
      if (model->subgraphs[subgraph_index]
              ->tensors[tensor_index]
              ->quantization) {
        // Remove unused min and max arrays from all tensors.
        model->subgraphs[subgraph_index]
            ->tensors[tensor_index]
            ->quantization->max.clear();
        model->subgraphs[subgraph_index]
            ->tensors[tensor_index]
            ->quantization->min.clear();

        // Shorten zero point arrays only on weight and bias tensors.
        if (model->subgraphs[subgraph_index]->tensors[tensor_index]->buffer) {
          if (model->subgraphs[subgraph_index]
                  ->tensors[tensor_index]
                  ->quantization->zero_point.empty()) {
            if (model->subgraphs[subgraph_index]
                    ->tensors[tensor_index]
                    ->quantization->zero_point.size() > 1) {
              model->subgraphs[subgraph_index]
                  ->tensors[tensor_index]
                  ->quantization->zero_point.resize(1);
            }
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc <= 1) {
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
  StripStrings(unpacked_model);
  RemoveExtraneousQuantizationData(unpacked_model);
  flatbuffers::FlatBufferBuilder fbb;
  auto optimized_model = tflite::Model::Pack(fbb, unpacked_model);
  fbb.Finish(optimized_model, tflite::ModelIdentifier());
  // flatbuffers::SaveFile(argv[2],
  //                       reinterpret_cast<char*>(fbb.GetBufferPointer()),
  //                       fbb.GetSize(), /*binary*/ true);
  int size = fbb.GetSize();
  fprintf(stderr, "New Optimized model size: %d\n", size);
  return 0;
}