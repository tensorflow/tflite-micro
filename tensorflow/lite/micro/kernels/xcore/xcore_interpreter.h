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

#ifndef XCORE_INTERPRETER_H_
#define XCORE_INTERPRETER_H_

#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_profiler.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

namespace tflite {
namespace micro {
namespace xcore {

/**
 * XCoreInterpreter class
 *
 * Implementation of the tflite::MicroInterpreter class, specialized
 * for XCore.
 */
class XCoreInterpreter : public tflite::MicroInterpreter {
 public:
  /**
   * Construct a customized XCoreInterpreter
   *
   * This constructor is for an application that provides specialized
   * implementations of memory loading and profiling.
   *
   * @param[in]  model       Pointer to a tflite::Model object
   * @param[in]  resolver    tflite::MicroOpResolver object with all required
   * operators registered.
   * @param[in]  allocator   Pointer to a tflite::MicroAllocator.  The allocator
   * must already be created with a memory buffer that is large enough to hold
   * the tensor arena.
   * @param[in]  reporter    Pointer to a tflite::ErrorReporter object.  If
   * null, no error reporting will be performed which reduces the code size and
   * runtime overhead. It is commmon to use null in release build targets.
   * @param[in]  dispatcher  Pointer to a Dispatcher object.
   * @param[in]  loader      Pointer to a MemoryLoader object.
   * @param[in]  profiler    (optional) Pointer to a tflite::MicroProfiler
   * object. If ommitted or null, no profiling will be performed which reduces
   * the code size and inference latency. It is commmon to omit in release build
   * targets.
   */
  XCoreInterpreter(const tflite::Model* model,
                   const tflite::MicroOpResolver& resolver,
                   tflite::MicroAllocator* allocator,
                   tflite::ErrorReporter* reporter, Dispatcher& dispatcher,
                   MemoryLoader& loader,
                   tflite::MicroProfiler* profiler = nullptr);

  /**
   * Construct the reference XCoreInterpreter implementation
   *
   * @param[in]  model       Pointer to a tflite::Model object
   * @param[in]  resolver    tflite::MicroOpResolver object with all required
   * operators registered.
   * @param[in]  arena       Pointer to a memory buffer for the tensor arena.
   * Must be large enough to store the arena.
   * @param[in]  arena_size  Size of the tensor arena in bytes.
   * @param[in]  reporter    Pointer to a tflite::ErrorReporter object.  If
   * null, no error reporting will be performed which reduces the code size and
   * runtime overhead. It is commmon to use null in release build targets.
   * @param[in]  profiler    (optional) Pointer to a XCoreProfiler object.
   * If ommitted or null, no profiling will be performed which reduces the code
   * size and inference latency. It is commmon to omit in release build targets.
   */
  XCoreInterpreter(const tflite::Model* model,
                   const tflite::MicroOpResolver& resolver, uint8_t* arena,
                   size_t arena_size, tflite::ErrorReporter* reporter,
                   XCoreProfiler* profiler = nullptr);

  /**
   * Construct the reference XCoreInterpreter implementation
   *
   * @param[in]  model       Pointer to a tflite::Model object
   * @param[in]  resolver    tflite::MicroOpResolver object with all required
   * operators registered.
   * @param[in]  allocator   Pointer to a tflite::MicroAllocator.  The allocator
   * must already be created with a memory buffer that is large enough to hold
   * the tensor arena.
   * @param[in]  reporter    Pointer to a tflite::ErrorReporter object.  If
   * null, no error reporting will be performed which reduces the code size and
   * runtime overhead. It is commmon to use null in release build targets.
   * @param[in]  profiler    (optional) Pointer to a XCoreProfiler object.
   * If ommitted or null, no profiling will be performed which reduces the code
   * size and inference latency. It is commmon to omit in release build targets.
   */
  XCoreInterpreter(const tflite::Model* model,
                   const tflite::MicroOpResolver& resolver,
                   tflite::MicroAllocator* allocator,
                   tflite::ErrorReporter* reporter,
                   XCoreProfiler* profiler = nullptr);

  /**
   * Get a tensor by index
   *
   * This implementation hides a method in the base class that performs a
   * persistent arena allocation for every call. Instead, this implementation
   * just returns a pointer to the underlying TfLiteTensor object, so it does
   * not explode the arena when the method is called repeatedly.
   *
   * @param[in]  tensor_index Index of the tensor to return.
   *
   * @return     Pointer to TfLiteTensor object
   */
  TfLiteTensor* tensor(size_t tensor_index);

  size_t tensors_size() const {
    const SubGraph* subgraph = model_->subgraphs()->Get(0);
    size_t length = subgraph->tensors()->Length();
    return length;
  }
};

}  // namespace xcore
}  // namespace micro
}  // namespace tflite

#endif  // XCORE_INTERPRETER_H_
