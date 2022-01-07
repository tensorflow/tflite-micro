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
#include "tensorflow/lite/micro/micro_allocator.h"
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

  // Allocate persistent buffer which has the same life time as the interpreter.
  // Returns nullptr on failure.
  // The memory is allocated from from tail.
  // This method is only available in Init or Prepare stage.
  // Virtual so that it can be mocked for kernel tests.
  // WARNING: This is an experimental interface that is subject to change.
  virtual void* AllocatePersistentBuffer(size_t bytes);

  // Request a scratch buffer in the arena through static memory planning.
  // This method is only available in Prepare stage and the buffer is allocated
  // by the interpreter between Prepare and Eval stage. In Eval stage,
  // GetScratchBuffer API can be used to fetch the address.
  // Virtual so that it can be mocked for kernel tests.
  // WARNING: This is an experimental interface that is subject to change.
  virtual TfLiteStatus RequestScratchBufferInArena(size_t bytes,
                                                   int* buffer_idx);

  // Get the scratch buffer pointer.
  // This method is only available in Eval stage.
  // Virtual so that it can be mocked for kernel tests.
  // WARNING: This is an experimental interface that is subject to change.
  virtual void* GetScratchBuffer(int buffer_idx);

  // Returns a TfLiteTensor struct for a given index.
  // Virtual so that it can be mocked for kernel tests.
  // WARNING: This is an experimental interface that is subject to change.
  virtual TfLiteTensor* GetTensor(int tensor_idx);

  // Returns a TfLiteEvalTensor struct for a given index.
  // Virtual so that it can be mocked for kernel tests.
  // WARNING: This is an experimental interface that is subject to change.
  virtual TfLiteEvalTensor* GetEvalTensor(int tensor_idx);

  // Requests that an error be reported with format string msg.
  void ReportOpError(const char* format, ...);

  // Accesses external contexts by type.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteExternalContext* GetExternalContext();

  // Sets the value of an external context. Does not take ownership of the
  // pointer.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus SetExternalContext(void* external_context_payload);

  // Returns the associated MicroGraph.
  // WARNING: This is an experimental interface that is subject to change.
  MicroGraph* GetGraph();

  // Sets the pointer to a list of ScratchBufferHandle instances.
  // Not API between TFLM and kernels. Primarily used by the framework for
  // housekeeping in MicroContext.
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
// these functions, and remove corresponding members from the TfLiteContext
// struct for TFLM.
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
inline void MicroContextReportOpError(struct TfLiteContext* context,
                                      const char* format, ...) {
  va_list args;
  va_start(args, format);
  GetMicroContext(context)->ReportOpError(format, args);
  va_end(args);
}
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_CONTEXT_H_
