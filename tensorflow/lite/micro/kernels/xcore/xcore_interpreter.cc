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

#include "tensorflow/lite/micro/kernels/xcore/xcore_interpreter.h"

#include "tensorflow/lite/micro/kernels/xcore/xcore_memory_loader.h"

namespace tflite {
namespace micro {
namespace xcore {

XCoreInterpreter::XCoreInterpreter(const tflite::Model* model,
                                   const tflite::MicroOpResolver& resolver,
                                   tflite::MicroAllocator* allocator,
                                   tflite::ErrorReporter* reporter,
                                   Dispatcher& dispatcher, MemoryLoader& loader,
                                   tflite::MicroProfiler* profiler)
    : tflite::MicroInterpreter(model, resolver, allocator, reporter, profiler) {
  SetDispatcher(&dispatcher);
  SetMemoryLoader(&loader);
}

XCoreInterpreter::XCoreInterpreter(const tflite::Model* model,
                                   const tflite::MicroOpResolver& resolver,
                                   tflite::MicroAllocator* allocator,
                                   tflite::ErrorReporter* reporter,
                                   XCoreProfiler* profiler)
    : tflite::MicroInterpreter(model, resolver, allocator, reporter, profiler) {
  auto* dispatcher_buf =
      allocator->AllocatePersistentBuffer(sizeof(GenericDispatcher));
  auto* dispatcher = new (dispatcher_buf) GenericDispatcher();
  SetDispatcher(dispatcher);

  auto* loader_buf =
      allocator->AllocatePersistentBuffer(sizeof(GenericMemoryLoader));
  auto* loader = new (loader_buf) GenericMemoryLoader();
  SetMemoryLoader(loader);

  if (profiler) {
    size_t max_event_count = 64;  // was operators_size()
    profiler->Init(allocator, max_event_count);
  }
}

XCoreInterpreter::XCoreInterpreter(const tflite::Model* model,
                                   const tflite::MicroOpResolver& resolver,
                                   uint8_t* arena, size_t arena_size,
                                   tflite::ErrorReporter* reporter,
                                   XCoreProfiler* profiler)
    : XCoreInterpreter::XCoreInterpreter(
          model, resolver, MicroAllocator::Create(arena, arena_size, reporter),
          reporter, profiler) {}

TfLiteTensor* XCoreInterpreter::tensor(size_t tensor_index) {
  auto ctx = context();
  return ctx.GetTensor(&ctx, tensor_index);
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite
