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

#ifndef XCORE_UTILS_H_
#define XCORE_UTILS_H_

#ifdef XCORE
#include <xs1.h>  // for XS1_RAM_BASE and XS1_RAM_SIZE
#endif

#include <cstdint>

#include "tensorflow/lite/micro/kernels/xcore/xcore_memory_loader.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

/* Unpack an integer data type from a byte array
 *  T  data type to unpack
 *
 * Example usage:
 *      int32_t t0 = unpack<int32_t>(&my_buffer[23]);
 *      int32_t t1 = unpack<int32_t>(&my_buffer[27]);
 */
template <class T>
T unpack(const uint8_t* buffer) {
  T retval = 0;
  for (int i = 0; i < sizeof(T); ++i) retval |= buffer[i] << (8 * i);
  return retval;
}

static inline bool is_ram_address(uintptr_t a) {
#ifdef XCORE
  return ((a >= XS1_RAM_BASE) && (a <= (XS1_RAM_BASE + XS1_RAM_SIZE)));
#else
  return true;
#endif
}

template <typename T>
inline size_t FetchBuffer(T** dest, T const* src, size_t size) {
  auto* loader = tflite::micro::xcore::GetMemoryLoader();
  return loader->Load((void**)dest, (void*)src, size);
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_UTILS_H_
