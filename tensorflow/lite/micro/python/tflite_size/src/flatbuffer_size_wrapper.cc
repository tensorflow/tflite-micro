/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "flatbuffer_size_wrapper.h"

#include <pybind11/pybind11.h>

#include "flatbuffer_size.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"

namespace tflite {

FlatbufferSizeWrapper::~FlatbufferSizeWrapper() {}

FlatbufferSizeWrapper::FlatbufferSizeWrapper() {}

std::string FlatbufferSizeWrapper::ConvertToJsonString(
    const char* in_flatbuffer) {
  std::string output = tflite::FlatBufferSizeToJsonString(
      reinterpret_cast<const uint8_t*>(in_flatbuffer),
      tflite::ModelTypeTable());

  return output;
}

}  // namespace tflite
