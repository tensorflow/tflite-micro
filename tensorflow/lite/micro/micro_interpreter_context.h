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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_CONTEXT_H_
#define TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_CONTEXT_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_interpreter_graph.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

// A full implementation of the MicroContext, to be used by the
// MicroInterpreter. Kernels should not depend on this directly. Instead they
// should only depend on the MicroContext.
class MicroInterpreterContext : public MicroContext {
 public:
  // Enum that allows MicroContext to keep track of the stages different memory
  // planning APIs are available to kernels.
  enum class InterpreterState {
    kInit,
    kPrepare,
    kMemoryPlanning,
    kInvoke,
  };

  // Does not take any ownership, and all pointers must refer to valid objects
  // that outlive the one constructed.
  MicroInterpreterContext(MicroAllocator* allocator, const Model* model,
                          MicroInterpreterGraph* graph);
  virtual ~MicroInterpreterContext();

  // Allocate persistent buffer which has the same life time as the interpreter.
  // Returns nullptr on failure.
  // The memory is allocated from the tail.
  // This method is only available in Init or Prepare stage.
  // Virtual so that it can be faked for kernel tests.
  virtual void* AllocatePersistentBuffer(size_t bytes) override;

  // Request a scratch buffer in the arena through static memory planning.
  // This method is only available in Prepare stage and the buffer is allocated
  // by the interpreter between Prepare and Eval stage. In Eval stage,
  // GetScratchBuffer API can be used to fetch the address.
  // Virtual so that it can be faked for kernel tests.
  virtual TfLiteStatus RequestScratchBufferInArena(size_t bytes,
                                                   int* buffer_idx) override;

  // Get the scratch buffer pointer.
  // This method is only available in Eval stage.
  // Virtual so that it can be faked for kernel tests.
  virtual void* GetScratchBuffer(int buffer_idx) override;

  // Returns a temporary TfLiteTensor struct for a given index.
  // Virtual so that it can be faked for kernel tests.
  virtual TfLiteTensor* AllocateTempTfLiteTensor(int tensor_idx) override;

  // Deallocates a temp TfLiteTensor.
  // Virtual so that it can be faked for kernel tests.
  virtual void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) override;

  // Returns a pointer to a temporary buffer (from the arena).
  // This API is only valid from the kernel's Prepare function and
  // the buffer's lifetime is also that of the Prepare function.
  // Virtual so that it can be faked for kernel tests.
  virtual uint8_t* AllocateTempBuffer(size_t size, size_t alignment) override;

  // Signals that the temporary buffer is no longer needed.
  // Virtual so that it can be faked for kernel tests.
  virtual void DeallocateTempBuffer(uint8_t* buffer) override;

  // Returns a TfLiteEvalTensor struct for a given index.
  // Virtual so that it can be faked for kernel tests.
  virtual TfLiteEvalTensor* GetEvalTensor(int tensor_idx) override;

  // Sets the State of MemoryPlanning MicroInterpreterContext
  void SetInterpreterState(InterpreterState state);

  // Sets the State of MemoryPlanning MicroInterpreterContext
  InterpreterState GetInterpreterState() const;

  // Does not take ownership of the pointer and the pointer must refer to valid
  // an object that outlive this class instance.
  // This can only be called once to set one external context.
  TfLiteStatus set_external_context(void* external_context_payload) override;

  void* external_context() override { return external_context_payload_; }

  MicroGraph& graph() override { return graph_; }

  // Sets the pointer to a list of ScratchBufferHandle instances.
  // Not API between TFLM and kernels. Primarily used by the framework for
  // housekeeping in MicroInterpreterContext.
  void SetScratchBufferHandles(ScratchBufferHandle* scratch_buffer_handles);

 private:
  MicroAllocator& allocator_;
  MicroInterpreterGraph& graph_;
  const Model* model_;
  InterpreterState state_;

  ScratchBufferHandle* scratch_buffer_handles_ = nullptr;
  void* external_context_payload_ = nullptr;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_CONTEXT_H_
