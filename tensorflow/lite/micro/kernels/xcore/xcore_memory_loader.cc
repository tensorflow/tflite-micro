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

#include "tensorflow/lite/micro/kernels/xcore/xcore_memory_loader.h"

#include <cassert>
#include <cstring>

#include "tensorflow/lite/micro/kernels/xcore/xcore_utils.h"

extern "C" {
#include "nn_operator.h"
}

namespace tflite {
namespace micro {
namespace xcore {

// global MemoryLoader shared by all operators
static MemoryLoader* kMemoryLoader = nullptr;

void SetMemoryLoader(MemoryLoader* loader) { kMemoryLoader = loader; }

MemoryLoader* GetMemoryLoader() {
  assert(kMemoryLoader);
  return kMemoryLoader;
}

size_t GenericMemoryLoader::Load(void** dest, const void* src, size_t size) {
  if (tflite::ops::micro::xcore::is_ram_address((uintptr_t)src)) {
    *dest = const_cast<void*>(src);
    return 0;
  } else if (size >= 128) {
    vpu_memcpy_ext(*dest, src, size);
  } else {
    memcpy(*dest, src, size);
  }
  return size;
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite
