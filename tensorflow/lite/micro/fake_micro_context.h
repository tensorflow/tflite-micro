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

#ifndef TENSORFLOW_LITE_MICRO_FAKE_MICRO_CONTEXT_H_
#define TENSORFLOW_LITE_MICRO_FAKE_MICRO_CONTEXT_H_

#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_graph.h"

namespace tflite {
// A fake of MicroContext for kernel util tests.
// TODO(b/272759060): FakeMicroContext currently inherits from MicroContext.
// Which allow tests to use functions from MicroContext that weren't added to
// FakeMicroContext in tests. This should be looked into further.

class FakeMicroContext : public MicroContext {
 public:
  ~FakeMicroContext() = default;

  FakeMicroContext(TfLiteTensor* tensors, SingleArenaBufferAllocator* allocator,
                   MicroGraph* micro_graph);

  void* AllocatePersistentBuffer(size_t bytes) override;
  TfLiteStatus RequestScratchBufferInArena(size_t bytes,
                                           int* buffer_index) override;
  void* GetScratchBuffer(int buffer_index) override;

  TfLiteTensor* AllocateTempTfLiteTensor(int tensor_index) override;
  void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) override;
  bool IsAllTempTfLiteTensorDeallocated();

  uint8_t* AllocateTempBuffer(size_t size, size_t alignment) override;
  void DeallocateTempBuffer(uint8_t* buffer) override;

  TfLiteEvalTensor* GetEvalTensor(int tensor_index) override;

  TfLiteStatus set_external_context(void* external_context_payload) override;
  void* external_context() override;
  MicroGraph& graph() override;

 private:
  static constexpr int kNumScratchBuffers_ = 12;

  MicroGraph& graph_;
  int scratch_buffer_count_ = 0;
  uint8_t* scratch_buffers_[kNumScratchBuffers_];

  TfLiteTensor* tensors_;
  int allocated_temp_count_ = 0;

  SingleArenaBufferAllocator* allocator_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_FAKE_MICRO_CONTEXT_H_
