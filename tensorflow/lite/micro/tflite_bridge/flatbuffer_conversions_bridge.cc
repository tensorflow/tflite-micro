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
#include "tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.h"

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace bridge {
// Converts the tensor data type used in the flat buffer to the representation
// used by the runtime.
TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type) {
  return ConvertTensorType(tensor_type, type, tflite::GetMicroErrorReporter());
}

// Wrapper function to wrap the parser function calls
TfLiteStatus CallBuiltinParseFunction(BuiltinParseFunction parser,
                                      const Operator* op,
                                      BuiltinDataAllocator* allocator,
                                      void** builtin_data) {
  return parser(op, tflite::GetMicroErrorReporter(), allocator, builtin_data);
}
}  // namespace bridge
}  // namespace tflite
