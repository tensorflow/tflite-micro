/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_PYTHON_COMPRESSION_UTILS_H_
#define TENSORFLOW_LITE_MICRO_PYTHON_COMPRESSION_UTILS_H_

#include <cstring>

#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Returns true if interpreter was built with compression support.
// When USE_TFLM_COMPRESSION is defined, this always returns true and
// the compiler can optimize away any if (!IsCompressionSupported()) branches.
inline constexpr bool IsCompressionSupported() {
#ifdef USE_TFLM_COMPRESSION
  return true;
#else
  return false;
#endif
}

// Helper to check if model has compression metadata.
// This is always compiled in, but when used with IsCompressionSupported()
// the entire check can be optimized away.
inline bool HasCompressionMetadata(const Model& model) {
  if (!model.metadata()) {
    return false;
  }

  for (size_t i = 0; i < model.metadata()->size(); ++i) {
    const auto* metadata = model.metadata()->Get(i);
    if (metadata && metadata->name() &&
        strcmp(metadata->name()->c_str(), "COMPRESSION_METADATA") == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_PYTHON_COMPRESSION_UTILS_H_