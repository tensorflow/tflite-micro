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

#ifndef XCORE_MEMORY_LOADER_H_
#define XCORE_MEMORY_LOADER_H_

#include <cstddef>

namespace tflite {
namespace micro {
namespace xcore {

/**
 * MemoryLoader abstract base class
 */
class MemoryLoader {
 public:
  MemoryLoader() = default;
  virtual ~MemoryLoader() = default;

  /**
   * Load memory from a potentially external memory segment.
   *
   * Override this method to provide an implementation specialized for
   * your application. For example, memcpy is not necessarily the most
   * efficient way to load data from flash.  An application developer
   * may prefer to read directly from flash using a flash library or
   * RTOS device driver. Additionally, in a multi-threaded application,
   * one may need to syncronize access to the memory.
   *
   *
   * @param[out] dest Pointer to the memory location to copy to
   * @param[in]  src  Pointer to the memory location to copy from
   * @param[in]  size Number of bytes to copy
   *
   * @return          Number of bytes loaded
   */
  virtual size_t Load(void** dest, const void* src, size_t size) = 0;
};

/**
 * GenericMemoryLoader class
 *
 * Reference implementation of the MemoryLoader abstract base class.
 * This class can be used to load data from SwMem (i.e. flash) or ExtMem (i.e.
 * LPDDR).
 */
class GenericMemoryLoader : public MemoryLoader {
 public:
  GenericMemoryLoader() = default;
  ~GenericMemoryLoader() = default;

  /**
   * Load memory from a potentially external memory segment.
   *
   * The reference implementation functions much like memcpy. The
   * primary difference is that this method may utilize the VPU to
   * accelerate copying memory.
   *
   * @param[out] dest Pointer to the memory location to copy to
   * @param[in]  src  Pointer to the memory location to copy from
   * @param[in]  size Number of bytes to copy
   *
   * @return          Number of bytes loaded
   */
  size_t Load(void** dest, const void* src, size_t size) override;
};

/**
 * Get the memory loader.
 *
 * @return    Global MemoryLoader object
 */
MemoryLoader* GetMemoryLoader();

/**
 * Set the memory loader.
 *
 * @param[in]  loader Pointer to the new global MemoryLoader object
 */
void SetMemoryLoader(MemoryLoader* loader);

}  // namespace xcore
}  // namespace micro
}  // namespace tflite

#endif  // XCORE_MEMORY_LOADER_H_
