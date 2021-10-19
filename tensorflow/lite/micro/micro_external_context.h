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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_EXTERNAL_CONTEXT_H_
#define TENSORFLOW_LITE_MICRO_MICRO_EXTERNAL_CONTEXT_H_

#include "tensorflow/lite/c/common.h"

// The list of external context types known to TF Lite Micro. This list exists
// solely to avoid conflicts and to ensure ops can share the external contexts
// they need.
typedef enum TfLiteMicroExternalContextSubType {
  kTfLiteMicroCadenceContext = 0,  // External context for Cadence accelerator.
  kTfLiteMicroMaxExternalContexts = 1
} TfLiteMicroExternalContextSubType;

// TfLite Micro external context definition when type in TfLiteExternalContext
// is kTfLiteMicroAcceleratorContext.
struct TfLiteMicroExternalContext : public TfLiteExternalContext {
  TfLiteMicroExternalContextSubType subtype;

  // Payload to external context. The kernel that uses this payload information
  // typecast it to the required struct based on subtype.
  void* context_data;
};

#endif  // TENSORFLOW_LITE_MICRO_MICRO_EXTERNAL_CONTEXT_H_
