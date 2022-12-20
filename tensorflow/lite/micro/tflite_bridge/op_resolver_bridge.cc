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

#include "tensorflow/lite/micro/tflite_bridge/op_resolver_bridge.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {

TfLiteStatus GetRegistrationFromOpCode(
    const OperatorCode* opcode, const OpResolver& op_resolver,
    const TfLiteRegistration** registration) {
  return GetRegistrationFromOpCode(
      opcode, op_resolver, tflite::GetMicroErrorReporter(), registration);
}
}  // namespace tflite
