/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef CODEGEN_RUNTIME_MICRO_CODEGEN_CONTEXT_H_
#define CODEGEN_RUNTIME_MICRO_CODEGEN_CONTEXT_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/micro/span.h"

namespace tflite {

struct Subgraph {
  Span<const size_t> inputs;
  Span<const size_t> outputs;
  Span<TfLiteNode> nodes;
  Span<TfLiteEvalTensor> tensors;
  TfLiteStatus (*invoke)(TfLiteContext*, Span<TfLiteNode>);
};

class MicroCodegenContext : public MicroContext, MicroGraph {
 public:
  MicroCodegenContext(TfLiteContext* context, Span<Subgraph> subgraphs);

  ~MicroCodegenContext() = default;

  // MicroContext API
  void* AllocatePersistentBuffer(size_t bytes) override;
  TfLiteStatus RequestScratchBufferInArena(size_t bytes,
                                           int* buffer_idx) override;
  void* GetScratchBuffer(int buffer_idx) override;
  TfLiteTensor* AllocateTempTfLiteTensor(int tensor_idx) override;
  void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) override;
  uint8_t* AllocateTempBuffer(size_t size, size_t alignment) override;
  void DeallocateTempBuffer(uint8_t* buffer) override;
  TfLiteEvalTensor* GetEvalTensor(int tensor_idx) override;
  TfLiteStatus set_external_context(void* external_context_payload) override;
  void* external_context() override;
  MicroGraph& graph() override;

  // MicroGraph API
  TfLiteStatus InvokeSubgraph(int subgraph_idx) override;
  size_t NumSubgraphInputs(int subgraph_idx) override;
  TfLiteEvalTensor* GetSubgraphInput(int subgraph_idx, int input_idx) override;
  size_t NumSubgraphOutputs(int subgraph_idx) override;
  TfLiteEvalTensor* GetSubgraphOutput(int subgraph_idx,
                                      int output_idx) override;
  int NumSubgraphs() override;
  MicroResourceVariables* GetResourceVariables() override;

 private:
  TfLiteContext* context_;
  Span<Subgraph> subgraphs_;
  size_t current_subgraph_idx_ = 0;
  void* external_context_payload_ = nullptr;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // CODEGEN_RUNTIME_MICRO_CODEGEN_CONTEXT_H_
