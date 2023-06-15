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

// This file is forked from TFLite's implementation in
// //depot/google3/third_party/tensorflow/lite/shared_library.h and contains a
// subset of it that's required by the TFLM interpreter. The Windows' ifdef is
// removed because TFLM doesn't support Windows.

#ifndef TENSORFLOW_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_SHARED_LIBRARY_H_
#define TENSORFLOW_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_SHARED_LIBRARY_H_

#include <dlfcn.h>

namespace tflite {

// SharedLibrary provides a uniform set of APIs across different platforms to
// handle dynamic library operations
class SharedLibrary {
 public:
  static inline void* GetSymbol(const char* symbol) {
    return dlsym(RTLD_DEFAULT, symbol);
  }
  static inline const char* GetError() { return dlerror(); }
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_SHARED_LIBRARY_H_
