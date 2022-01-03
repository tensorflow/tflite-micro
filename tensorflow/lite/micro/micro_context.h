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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_CONTEXT_H_
#define TENSORFLOW_LITE_MICRO_MICRO_CONTEXT_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_graph.h"

namespace tflite {
// MicroContext is eventually going to become the API between TFLM and the
// kernels, replacing all the functions in TfLiteContext. The end state is code
// kernels to have code like:
//
// MicroContext* micro_context = GetMicroContext(context);
// micro_context-><TFLM kernel API>
class MicroContext {
 public:
  // Does not take any ownership, and all pointers must refer to valid objects
  // that outlive the one constructed.
  explicit MicroContext(MicroAllocator* allocator, const Model* model,
                        MicroGraph* graph);

  // Functions that will be assigned to function pointers on TfLiteContext:
  static void* AllocatePersistentBuffer(TfLiteContext* ctx, size_t bytes);
  static TfLiteStatus RequestScratchBufferInArena(TfLiteContext* ctx,
                                                  size_t bytes,
                                                  int* buffer_idx);
  static void* GetScratchBuffer(TfLiteContext* ctx, int buffer_idx);

  static TfLiteTensor* GetTensor(const struct TfLiteContext* context,
                                 int tensor_idx);
  static TfLiteEvalTensor* GetEvalTensor(const struct TfLiteContext* context,
                                         int tensor_idx);
  static void ReportOpError(struct TfLiteContext* context, const char* format,
                            ...);
  static TfLiteExternalContext* GetExternalContext(
      TfLiteContext* context, TfLiteExternalContextType unused);
  TfLiteStatus SetExternalContext(void* external_context_payload);

  // Functions that provide unique API between TFLM and kernels.
  // Virtual so that it can be mocked.
  virtual MicroGraph* GetGraph();

  // Sets the pointer to a list of ScratchBufferHandle instances. This is used
  // by the framework only.
  void SetScratchBufferHandles(ScratchBufferHandle* scratch_buffer_handles);

 private:
  MicroAllocator& allocator_;
  const Model* model_;
  MicroGraph* graph_;
  ScratchBufferHandle* scratch_buffer_handles_ = nullptr;
  void* external_context_payload_ = nullptr;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

inline MicroContext* GetMicroContext(const struct TfLiteContext* context) {
  return reinterpret_cast<MicroContext*>(context->impl_);
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_CONTEXT_H_
