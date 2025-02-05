/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_H_
#define TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_H_

#include <cstddef>
#include <cstdint>

#ifdef USE_TFLM_COMPRESSION

#include <initializer_list>

#endif  // USE_TFLM_COMPRESSION

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter_context.h"
#include "tensorflow/lite/micro/micro_interpreter_graph.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/schema/schema_generated.h"

/// Copied from tensorflow/lite/version.h to avoid a dependency chain into
// tensorflow/core.
#define TFLITE_SCHEMA_VERSION (3)

namespace tflite {

class MicroInterpreter {
 public:
  // The lifetime of the model, op resolver, tensor arena, error reporter,
  // resource variables, and profiler must be at least as long as that of the
  // interpreter object, since the interpreter may need to access them at any
  // time. This means that you should usually create them with the same scope as
  // each other, for example having them all allocated on the stack as local
  // variables through a top-level function. The interpreter doesn't do any
  // deallocation of any of the pointed-to objects, ownership remains with the
  // caller.
  MicroInterpreter(const Model* model, const MicroOpResolver& op_resolver,
                   uint8_t* tensor_arena, size_t tensor_arena_size,
                   MicroResourceVariables* resource_variables = nullptr,
                   MicroProfilerInterface* profiler = nullptr,
                   bool preserve_all_tensors = false);

  // Create an interpreter instance using an existing MicroAllocator instance.
  // This constructor should be used when creating an allocator that needs to
  // have allocation handled in more than one interpreter or for recording
  // allocations inside the interpreter. The lifetime of the allocator must be
  // as long as that of the interpreter object.
  MicroInterpreter(const Model* model, const MicroOpResolver& op_resolver,
                   MicroAllocator* allocator,
                   MicroResourceVariables* resource_variables = nullptr,
                   MicroProfilerInterface* profiler = nullptr);

  ~MicroInterpreter();

  // Runs through the model and allocates all necessary input, output and
  // intermediate tensors.
  TfLiteStatus AllocateTensors();

  // In order to support partial graph runs for strided models, this can return
  // values other than kTfLiteOk and kTfLiteError.
  // TODO(b/149795762): Add this to the TfLiteStatus enum.
  TfLiteStatus Invoke();

  // This is the recommended API for an application to pass an external payload
  // pointer as an external context to kernels. The life time of the payload
  // pointer should be at least as long as this interpreter. TFLM supports only
  // one external context.
  TfLiteStatus SetMicroExternalContext(void* external_context_payload);

  TfLiteTensor* input(size_t index);
  size_t inputs_size() const {
    return model_->subgraphs()->Get(0)->inputs()->size();
  }
  const flatbuffers::Vector<int32_t>& inputs() const {
    return *model_->subgraphs()->Get(0)->inputs();
  }
  TfLiteTensor* input_tensor(size_t index) { return input(index); }
  template <class T>
  T* typed_input_tensor(int tensor_index) {
    if (TfLiteTensor* tensor_ptr = input_tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return GetTensorData<T>(tensor_ptr);
      }
    }
    return nullptr;
  }

  TfLiteTensor* output(size_t index);
  size_t outputs_size() const {
    return model_->subgraphs()->Get(0)->outputs()->size();
  }
  const flatbuffers::Vector<int32_t>& outputs() const {
    return *model_->subgraphs()->Get(0)->outputs();
  }
  TfLiteTensor* output_tensor(size_t index) { return output(index); }
  template <class T>
  T* typed_output_tensor(int tensor_index) {
    if (TfLiteTensor* tensor_ptr = output_tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return GetTensorData<T>(tensor_ptr);
      }
    }
    return nullptr;
  }

  // Returns a pointer to the tensor for the corresponding tensor_index
  TfLiteEvalTensor* GetTensor(int tensor_index, int subgraph_index = 0);

  // Reset the state to be what you would expect when the interpreter is first
  // created. i.e. after Init and Prepare is called for the very first time.
  TfLiteStatus Reset();

  TfLiteStatus initialization_status() const { return initialization_status_; }

  // Populates node and registration pointers representing the inference graph
  // of the model from values inside the flatbuffer (loaded from the TfLiteModel
  // instance). Persistent data (e.g. operator data) is allocated from the
  // arena.
  TfLiteStatus PrepareNodeAndRegistrationDataFromFlatbuffer();

  // For debugging only.
  // Returns the actual used arena in bytes. This method gives the optimal arena
  // size. It's only available after `AllocateTensors` has been called.
  // Note that normally `tensor_arena` requires 16 bytes alignment to fully
  // utilize the space. If it's not the case, the optimial arena size would be
  // arena_used_bytes() + 16.
  size_t arena_used_bytes() const { return allocator_.used_bytes(); }

  // Returns True if all Tensors are being preserves
  // TODO(b/297106074) : revisit making C++ example or test for
  // preserve_all_tesnors
  bool preserve_all_tensors() const {
    return allocator_.preserves_all_tensor();
  }

  // Set the alternate MicroProfilerInterface.
  // This value is passed through to the MicroContext.
  // This can be used to profile subsystems simultaneously with the profiling
  // of kernels during the Eval phase.  See (b/379584353).
  // The alternate MicroProfilerInterface is currently used by the tensor
  // decompression subsystem.
  TfLiteStatus SetAlternateProfiler(MicroProfilerInterface* alt_profiler);

#ifdef USE_TFLM_COMPRESSION

  // Set the alternate decompression memory regions.
  // Can only be called during the MicroInterpreter kInit state (i.e. must
  // be called before MicroInterpreter::AllocateTensors).
  TfLiteStatus SetDecompressionMemory(
      const std::initializer_list<MicroContext::AlternateMemoryRegion>&
          regions);

#endif  // USE_TFLM_COMPRESSION

 protected:
  const MicroAllocator& allocator() const { return allocator_; }
  const TfLiteContext& context() const { return context_; }

 private:
  // TODO(b/158263161): Consider switching to Create() function to enable better
  // error reporting during initialization.
  void Init(MicroProfilerInterface* profiler);

  // Gets the current subgraph index used from within context methods.
  int get_subgraph_index() { return graph_.GetCurrentSubgraphIndex(); }

  const Model* model_;
  const MicroOpResolver& op_resolver_;
  TfLiteContext context_ = {};
  MicroAllocator& allocator_;
  MicroInterpreterGraph graph_;
  bool tensors_allocated_;

  TfLiteStatus initialization_status_;

  ScratchBufferHandle* scratch_buffer_handles_ = nullptr;

  // TODO(b/162311891): Clean these pointers up when this class supports buffers
  // from TfLiteEvalTensor.
  TfLiteTensor** input_tensors_;
  TfLiteTensor** output_tensors_;

  MicroInterpreterContext micro_context_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_H_
