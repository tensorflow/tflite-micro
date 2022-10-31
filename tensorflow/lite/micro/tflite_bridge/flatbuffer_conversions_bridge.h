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
#ifndef TENSORFLOW_LITE_MICRO_FLATBUFFER_CONVERSIONS_BRIDGE_H_
#define TENSORFLOW_LITE_MICRO_FLATBUFFER_CONVERSIONS_BRIDGE_H_

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"

namespace tflite {

// Forward declaration of the classes, structs and enums being reffered here
enum TensorType : int8_t;
class ErrorReporter;
struct Operator;

// namespace bridge to wrap the TFLite API and features for use in TFLM
namespace bridge {

// Using declarative to create tflite::bridge::BuiltinDataAllocator from
// tflite::BuiltinDataAllocator
using BuiltinDataAllocator = BuiltinDataAllocator;

// Using declarative for OpParserFunction pointer
using BuiltinParseFunction = TfLiteStatus (*)(const Operator* op,
                                              ErrorReporter* error_reporter,
                                              BuiltinDataAllocator* allocator,
                                              void** builtin_data);

// Converts the tensor data type used in the flat buffer to the representation
// used by the runtime.
TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type);

// CallBuiltinParseFunction is a wrapper function to wrap the parser function
// calls
TfLiteStatus CallBuiltinParseFunction(BuiltinParseFunction parser,
                                      const Operator* op,
                                      BuiltinDataAllocator* allocator,
                                      void** builtin_data);
}  // namespace bridge
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_FLATBUFFER_CONVERSIONS_BRIDGE_H_
