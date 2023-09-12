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

#include "codegen/runtime/micro_codegen_context.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

MicroCodegenContext::MicroCodegenContext(TfLiteContext* context,
                                         Span<Subgraph> subgraphs)
    : context_(context), subgraphs_(subgraphs) {}

void* MicroCodegenContext::GetScratchBuffer(int buffer_idx) {
  // TODO(rjascani): Implement scratch buffers
  return nullptr;
}

TfLiteEvalTensor* MicroCodegenContext::GetEvalTensor(int tensor_idx) {
  TFLITE_DCHECK(static_cast<size_t>(tensor_idx) <
                subgraphs_[current_subgraph_idx_].tensors.size());
  return &subgraphs_[current_subgraph_idx_].tensors[tensor_idx];
}

TfLiteStatus MicroCodegenContext::set_external_context(
    void* external_context_payload) {
  if (external_context_payload == nullptr ||
      external_context_payload_ != nullptr) {
    MicroPrintf(
        "Attempting to set external context to %x but it was %x already",
        external_context_payload, external_context_payload_);
    return kTfLiteError;
  }

  external_context_payload_ = external_context_payload;
  return kTfLiteOk;
}

void* MicroCodegenContext::external_context() {
  return external_context_payload_;
}

MicroGraph& MicroCodegenContext::graph() { return *this; }

void* MicroCodegenContext::AllocatePersistentBuffer(size_t) {
  // Not allowed at Eval
  TFLITE_ABORT;
  return nullptr;
}

TfLiteStatus MicroCodegenContext::RequestScratchBufferInArena(size_t, int*) {
  // Not allowed at Eval
  TFLITE_ABORT;
  return kTfLiteError;
}

TfLiteTensor* MicroCodegenContext::AllocateTempTfLiteTensor(int) {
  // Not allowed at Eval
  TFLITE_ABORT;
  return nullptr;
}

void MicroCodegenContext::DeallocateTempTfLiteTensor(TfLiteTensor*) {
  // Not allowed at Eval
  TFLITE_ABORT;
}

uint8_t* MicroCodegenContext::AllocateTempBuffer(size_t, size_t) {
  // Not allowed at Eval
  TFLITE_ABORT;
  return nullptr;
}

void MicroCodegenContext::DeallocateTempBuffer(uint8_t*) {
  // Not allowed at Eval
  TFLITE_ABORT;
}

TfLiteStatus MicroCodegenContext::InvokeSubgraph(int subgraph_idx) {
  TF_LITE_ENSURE(context_,
                 static_cast<size_t>(subgraph_idx) < subgraphs_.size());
  size_t previous_subgraph_idx = current_subgraph_idx_;
  current_subgraph_idx_ = subgraph_idx;
  TfLiteStatus status =
      subgraphs_[subgraph_idx].invoke(context_, subgraphs_[subgraph_idx].nodes);
  current_subgraph_idx_ = previous_subgraph_idx;
  return status;
}

size_t MicroCodegenContext::NumSubgraphInputs(int subgraph_idx) {
  TFLITE_DCHECK(static_cast<size_t>(subgraph_idx) < subgraphs_.size());
  return subgraphs_[subgraph_idx].inputs.size();
}

TfLiteEvalTensor* MicroCodegenContext::GetSubgraphInput(int subgraph_idx,
                                                        int input_idx) {
  TFLITE_DCHECK(static_cast<size_t>(subgraph_idx) < subgraphs_.size());
  TFLITE_DCHECK(static_cast<size_t>(input_idx) <
                subgraphs_[subgraph_idx].inputs.size());
  const size_t tensor_idx = subgraphs_[subgraph_idx].inputs[input_idx];
  return &subgraphs_[subgraph_idx].tensors[tensor_idx];
}

size_t MicroCodegenContext::NumSubgraphOutputs(int subgraph_idx) {
  TFLITE_DCHECK(static_cast<size_t>(subgraph_idx) < subgraphs_.size());
  return subgraphs_[subgraph_idx].outputs.size();
}

TfLiteEvalTensor* MicroCodegenContext::GetSubgraphOutput(int subgraph_idx,
                                                         int output_idx) {
  TFLITE_DCHECK(static_cast<size_t>(subgraph_idx) < subgraphs_.size());
  TFLITE_DCHECK(static_cast<size_t>(output_idx) <
                subgraphs_[subgraph_idx].outputs.size());
  const size_t tensor_idx = subgraphs_[subgraph_idx].outputs[output_idx];
  return &subgraphs_[subgraph_idx].tensors[tensor_idx];
}

int MicroCodegenContext::NumSubgraphs() { return subgraphs_.size(); }

MicroResourceVariables* MicroCodegenContext::GetResourceVariables() {
  return nullptr;
}

}  // namespace tflite
