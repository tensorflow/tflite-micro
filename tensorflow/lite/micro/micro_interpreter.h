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

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/schema/schema_generated.h"

/// Copied from tensorflow/lite/version.h to avoid a dependency chain into
// tensorflow/core.
#define TFLITE_SCHEMA_VERSION (3)

namespace tflite {

// Encapsulates a pre-trained model and drives the model inference.
//
//
// @note This class is not thread-safe.
// The client is responsible for ensuring serialized interaction to avoid data
// races and undefined behavior.
class MicroInterpreter {
 public:
  // Constructor. Creates an instance with an allocated tensor arena.
  //
  // The lifetime of the model, op resolver, tensor arena, error reporter,
  // resource variables, and profiler must be at least as long as that of the
  // interpreter object, since the interpreter may need to access them at any
  // time. This means that you should usually create them with the same scope as
  // each other, for example having them all allocated on the stack as local
  // variables through a top-level function. The interpreter doesn't do any
  // deallocation of any of the pointed-to objects, ownership remains with the
  // caller.
  //
  // @param model A trained TensorFlow Lite model.
  // @param op_resolver The op resolver that contains all ops used by the model.
  //   This is usually an instance of `tflite::MicroMutableOpResolver`.
  // @param tensor_arena The allocated memory for all intermediate tensor data.
  // @param tensor_arena_size The size of `tensor_arena`.
  // @param resource_variables Handles assign/read ops for resource variables.
  //   See [Resource variable docs](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/resource_variables.md).
  // @param profiler Handles profiling for op kernels and TFLM routines.
  //   See [Profiling docs](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/profiling.md).
  MicroInterpreter(const Model* model, const MicroOpResolver& op_resolver,
                   uint8_t* tensor_arena, size_t tensor_arena_size,
                   MicroResourceVariables* resource_variables = nullptr,
                   MicroProfilerInterface* profiler = nullptr);

  // Constructor. Creates an instance using an existing MicroAllocator instance.
  //
  // This constructor should be used when creating an allocator that needs to
  // have allocation handled in more than one interpreter or for recording
  // allocations inside the interpreter. The lifetime of the allocator must be
  // as long as that of the interpreter object.
  //
  // @param model A trained TensorFlow Lite model.
  // @param op_resolver The op resolver that contains all ops used by the model.
  //   This is usually an instance of `tflite::MicroMutableOpResolver`.
  // @param allocator The object that allocates all intermediate tensor data.
  // @param resource_variables Handles assign/read ops for resource variables.
  //   See [Resource variable docs](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/resource_variables.md).
  // @param profiler Handles profiling for op kernels and TFLM routines.
  //   See [Profiling docs](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/profiling.md).
  MicroInterpreter(const Model* model, const MicroOpResolver& op_resolver,
                   MicroAllocator* allocator,
                   MicroResourceVariables* resource_variables = nullptr,
                   MicroProfilerInterface* profiler = nullptr);

  ~MicroInterpreter();

  // Allocates all the model's necessary input, output and intermediate tensors.
  //
  // This will redim dependent tensors using the input tensor dimensionality as
  // given. This is relatively expensive. This must be called after the
  // interpreter has been created and before running inference (and accessing
  // tensor buffers), and must be called again if (and only if) an input tensor
  // is resized.
  //
  // @return Atatus of success or failure. Will fail if any of the
  // ops in the model (other than those which were rewritten by delegates, if
  // any) are not supported by the Interpreter's OpResolver.
  TfLiteStatus AllocateTensors();

  // Invokes the model to run inference using allocated input tensors.
  //
  // In order to support partial graph runs for strided models, this can return
  // values other than kTfLiteOk and kTfLiteError.
  TfLiteStatus Invoke();

  // This is the recommended API for an application to pass an external payload
  // pointer as an external context to kernels. The life time of the payload
  // pointer should be at least as long as this interpreter. TFLM supports only
  // one external context.
  TfLiteStatus SetMicroExternalContext(void* external_context_payload);

  // Gets a mutable pointer to an input tensor.
  // @param index The index position of the input tensor.
  //   Must be between 0 and `inputs_size()`.
  // @return The input tensor.
  TfLiteTensor* input(size_t index);

  // Gets the size of the input tensors.
  size_t inputs_size() const {
    return model_->subgraphs()->Get(0)->inputs()->size();
  }

  // Gets a read-only list of all inputs.
  const flatbuffers::Vector<int32_t>& inputs() const {
    return *model_->subgraphs()->Get(0)->inputs();
  }

  // Same as `input()`.
  TfLiteTensor* input_tensor(size_t index) { return input(index); }

  // Gets a mutable pointer into the data of a given input tensor.
  //
  // The given index must be between 0 and `inputs_size()`.
  template <class T>
  T* typed_input_tensor(int tensor_index) {
    if (TfLiteTensor* tensor_ptr = input_tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return GetTensorData<T>(tensor_ptr);
      }
    }
    return nullptr;
  }

  // Gets a mutable pointer to an output tensor.
  // @param index The index position of the output tensor.
  //   Must be between 0 and `outputs_size()`.
  // @return The output tensor.
  TfLiteTensor* output(size_t index);

  // Gets the size of the output tensors.
  size_t outputs_size() const {
    return model_->subgraphs()->Get(0)->outputs()->size();
  }

  // Gets a read-only list of all outputs.
  const flatbuffers::Vector<int32_t>& outputs() const {
    return *model_->subgraphs()->Get(0)->outputs();
  }

  // Same as `output()`.
  TfLiteTensor* output_tensor(size_t index) { return output(index); }

  // Gets a mutable pointer into the data of a given output tensor.
  //
  // The given index must be between 0 and `outputs_size()`.
  template <class T>
  T* typed_output_tensor(int tensor_index) {
    if (TfLiteTensor* tensor_ptr = output_tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return GetTensorData<T>(tensor_ptr);
      }
    }
    return nullptr;
  }

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
  MicroGraph graph_;
  bool tensors_allocated_;

  TfLiteStatus initialization_status_;

  ScratchBufferHandle* scratch_buffer_handles_ = nullptr;

  // TODO(b/162311891): Clean these pointers up when this class supports buffers
  // from TfLiteEvalTensor.
  TfLiteTensor** input_tensors_;
  TfLiteTensor** output_tensors_;

  MicroContext micro_context_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_H_
