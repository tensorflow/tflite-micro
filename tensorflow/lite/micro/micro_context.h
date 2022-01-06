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

  // Functions that provide unique API between TFLM and kernels.
  // Virtual so that they can be mocked.
  virtual void* AllocatePersistentBuffer(size_t bytes);
  virtual TfLiteStatus RequestScratchBufferInArena(size_t bytes,
                                                   int* buffer_idx);
  virtual void* GetScratchBuffer(int buffer_idx);

  virtual TfLiteTensor* GetTensor(int tensor_idx);
  virtual TfLiteEvalTensor* GetEvalTensor(int tensor_idx);

  virtual void ReportOpError(const char* format, ...);

  virtual TfLiteExternalContext* GetExternalContext();
  virtual TfLiteStatus SetExternalContext(void* external_context_payload);

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

// Deprecated API. Prefer to using the MicroContext API directly from the
// kernels.
// TODO(b/213010668): migrate all existing kernels to use MicroContext, delete
// this function, and remove GetTensor from the TfLiteContext struct for TFLM.
inline void* MicroContextAllocatePersistentBuffer(TfLiteContext* ctx,
                                                  size_t bytes) {
  return GetMicroContext(ctx)->AllocatePersistentBuffer(bytes);
}
inline TfLiteStatus MicroContextRequestScratchBufferInArena(TfLiteContext* ctx,
                                                            size_t bytes,
                                                            int* buffer_idx) {
  return GetMicroContext(ctx)->RequestScratchBufferInArena(bytes, buffer_idx);
}
inline void* MicroContextGetScratchBuffer(TfLiteContext* ctx, int buffer_idx) {
  return GetMicroContext(ctx)->GetScratchBuffer(buffer_idx);
}
inline TfLiteTensor* MicroContextGetTensor(const struct TfLiteContext* context,
                                           int tensor_idx) {
  return GetMicroContext(context)->GetTensor(tensor_idx);
}
inline TfLiteEvalTensor* MicroContextGetEvalTensor(
    const struct TfLiteContext* context, int tensor_idx) {
  return GetMicroContext(context)->GetEvalTensor(tensor_idx);
}
inline TfLiteExternalContext* MicroContextGetExternalContext(
    TfLiteContext* context, TfLiteExternalContextType unused) {
  return GetMicroContext(context)->GetExternalContext();
}
inline void ReportOpError(struct TfLiteContext* context, const char* format,
                          ...) {
  va_list args;
  va_start(args, format);
  GetMicroContext(context)->ReportOpError(format, args);
  va_end(args);
}
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_CONTEXT_H_
