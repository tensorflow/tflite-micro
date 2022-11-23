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
#ifndef TENSORFLOW_LITE_MICRO_TFLITE_BRIDGE_OP_RESOLVER_BRIDGE_H_
#define TENSORFLOW_LITE_MICRO_TFLITE_BRIDGE_OP_RESOLVER_BRIDGE_H_

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"  // needed for the Using declarative

namespace tflite {

// Forward declaration of the classes and structs used here.
struct OperatorCode;

using TfLiteBridgeOpResolver = OpResolver;

// Handles the logic for converting between an OperatorCode structure extracted
// from a flatbuffer and information about a registered operator
// implementation.
TfLiteStatus GetRegistrationFromOpCode(const OperatorCode* opcode,
                                       const OpResolver& op_resolver,
                                       const TfLiteRegistration** registration);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TFLITE_BRIDGE_OP_RESOLVER_BRIDGE_H_
