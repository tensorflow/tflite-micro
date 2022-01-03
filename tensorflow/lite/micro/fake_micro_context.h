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

class FakeMicroContext : public MicroContext {
 public:
  FakeMicroContext(TfLiteTensor* tensors, SimpleMemoryAllocator* allocator,
                   MicroGraph* micro_graph);

  // Overload
  static TfLiteTensor* GetTensor(const struct TfLiteContext* context,
                                 int tensor_index);
  static TfLiteEvalTensor* GetEvalTensor(const struct TfLiteContext* context,
                                         int tensor_index);
  static void* AllocatePersistentBuffer(TfLiteContext* context, size_t bytes);
  static TfLiteStatus RequestScratchBufferInArena(TfLiteContext* context,
                                                  size_t bytes,
                                                  int* buffer_index);
  static void* GetScratchBuffer(TfLiteContext* context, int buffer_index);
  static void ReportOpError(struct TfLiteContext* context, const char* format,
                            ...);

  MicroGraph* GetGraph() override;

 private:
  static FakeMicroContext* GetMicroContext(const struct TfLiteContext* context);

  static constexpr int kNumScratchBuffers_ = 12;

  int scratch_buffer_count_ = 0;
  uint8_t* scratch_buffers_[kNumScratchBuffers_];

  TfLiteTensor* tensors_;
  SimpleMemoryAllocator* allocator_;
  MicroGraph* micro_graph_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_FAKE_MICRO_CONTEXT_H_
