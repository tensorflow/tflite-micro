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

#include <stddef.h>

#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {

struct OpData {
  int then_subgraph_index;
  int else_subgraph_index;
  void* intermediate_input_buffer;
  void* intermediate_output_buffer;
};

enum class CopyInputDirection {
  kFromIfToIntermediate,
  kFromIntermediateToSubgraph,
  kFromIntermediateToIf,
};

enum class CopyOutputDirection {
  kFromSubgraphToIntermediate,
  kFromIntermediateToIf,
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus AllocateIntermediateBuffer(TfLiteContext* context,
                                        TfLiteNode* node) {
  // The first input is the condition.
  constexpr int kInputTensorIndexBasis = 1;
  size_t num_inputs = node->inputs->size - kInputTensorIndexBasis;
  size_t num_outputs = node->outputs->size;
  TfLiteTensor* input_data_tensor;

  size_t total_input_bytes = 0;
  MicroContext* micro_context = GetMicroContext(context);
  for (size_t i = 0; i < num_inputs; i++) {
    input_data_tensor = micro_context->AllocateTempInputTensor(
        node, kInputTensorIndexBasis + i);
    TF_LITE_ENSURE(context, input_data_tensor != nullptr);
    size_t type_size;
    TF_LITE_ENSURE_STATUS(
        TfLiteTypeSizeOf(input_data_tensor->type, &type_size));
    total_input_bytes += NumElements(input_data_tensor) * type_size;

    micro_context->DeallocateTempTfLiteTensor(input_data_tensor);
  }
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  op_data->intermediate_input_buffer =
      context->AllocatePersistentBuffer(context, total_input_bytes);

  size_t total_output_bytes = 0;
  TfLiteTensor* output_data_tensor;
  for (size_t i = 0; i < num_outputs; i++) {
    output_data_tensor = micro_context->AllocateTempOutputTensor(node, i);

    size_t type_size;
    TF_LITE_ENSURE_STATUS(
        TfLiteTypeSizeOf(output_data_tensor->type, &type_size));
    total_output_bytes += NumElements(output_data_tensor) * type_size;
    micro_context->DeallocateTempTfLiteTensor(output_data_tensor);
  }
  op_data->intermediate_output_buffer =
      context->AllocatePersistentBuffer(context, total_output_bytes);
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const auto* params =
      reinterpret_cast<const TfLiteIfParams*>(node->builtin_data);
  op_data->then_subgraph_index = params->then_subgraph_index;
  op_data->else_subgraph_index = params->else_subgraph_index;

  TF_LITE_ENSURE(context, node->inputs->size > 0);

  // The first input is the condition.
  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
  TfLiteTensor* cond = micro_context->AllocateTempInputTensor(node, 0);

  TF_LITE_ENSURE(context, cond != nullptr);
  TF_LITE_ENSURE_EQ(context, cond->type, kTfLiteBool);
  TF_LITE_ENSURE_EQ(context, NumElements(cond), 1);

  micro_context->DeallocateTempTfLiteTensor(cond);

  // The first input of the node is the condition. The rest of inputs are
  // passed to the branch subgraphs. Therefore, the number of subgraph inputs
  // will be the number of node inputs - 1.
  size_t num_inputs = node->inputs->size - 1;
  size_t num_outputs = node->outputs->size;

  MicroGraph& graph_info = micro_context->graph();

  TF_LITE_ENSURE(context,
                 op_data->then_subgraph_index < graph_info.NumSubgraphs());
  TF_LITE_ENSURE(context,
                 op_data->else_subgraph_index < graph_info.NumSubgraphs());

  TF_LITE_ENSURE_EQ(context, num_inputs,
                    graph_info.NumSubgraphInputs(op_data->then_subgraph_index));
  TF_LITE_ENSURE_EQ(
      context, num_outputs,
      graph_info.NumSubgraphOutputs(op_data->then_subgraph_index));

  TF_LITE_ENSURE_OK(context, AllocateIntermediateBuffer(context, node));

  return kTfLiteOk;
}

TfLiteStatus CopyOutputBetweenIfAndSubgraph(
    TfLiteContext* context, TfLiteNode* node, MicroGraph& graph_info,
    int active_branch_subgraph_index, CopyOutputDirection copy_direction) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  int8_t* intermediate_copy_buffer =
      reinterpret_cast<int8_t*>(op_data->intermediate_output_buffer);
  for (size_t i = 0;
       i < graph_info.NumSubgraphOutputs(active_branch_subgraph_index); ++i) {
    const TfLiteEvalTensor* output =
        tflite::micro::GetEvalOutput(context, node, i);

    TfLiteEvalTensor* subgraph_output =
        graph_info.GetSubgraphOutput(active_branch_subgraph_index, i);

    // These checks must occur in Eval since TfLiteEvalTensors are not available
    // during Prepare.
    size_t output_bytes;
    size_t subgraph_output_bytes;
    TF_LITE_ENSURE_OK(context,
                      TfLiteEvalTensorByteLength(output, &output_bytes));
    TF_LITE_ENSURE_OK(context, TfLiteEvalTensorByteLength(
                                   subgraph_output, &subgraph_output_bytes));
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, subgraph_output->type);
    TF_LITE_ENSURE_EQ(context, output_bytes, subgraph_output_bytes);

    void* source;
    void* destination;
    switch (copy_direction) {
      case CopyOutputDirection::kFromSubgraphToIntermediate:
        destination = intermediate_copy_buffer;
        source = subgraph_output->data.raw;
        break;
      case CopyOutputDirection::kFromIntermediateToIf:
        destination = output->data.raw;
        source = intermediate_copy_buffer;
        break;
        // NO default. enum is owned by this file and new enum shall be handled.
    }
    memcpy(destination, source, output_bytes);
    intermediate_copy_buffer += output_bytes;
  }
  return kTfLiteOk;
}

TfLiteStatus CopyInputBetweenIfAndSubgraph(TfLiteContext* context,
                                           TfLiteNode* node,
                                           MicroGraph& graph_info,
                                           int active_branch_subgraph_index,
                                           CopyInputDirection copy_direction) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  int8_t* intermediate_copy_buffer =
      reinterpret_cast<int8_t*>(op_data->intermediate_input_buffer);
  for (size_t i = 0;
       i < graph_info.NumSubgraphInputs(active_branch_subgraph_index); ++i) {
    const TfLiteEvalTensor* input =
        tflite::micro::GetEvalInput(context, node, i + 1);

    TfLiteEvalTensor* subgraph_input =
        graph_info.GetSubgraphInput(active_branch_subgraph_index, i);

    // These checks must occur in Eval since TfLiteEvalTensors are not available
    // during Prepare.
    size_t input_bytes;
    size_t subgraph_input_bytes;
    TF_LITE_ENSURE_OK(context, TfLiteEvalTensorByteLength(input, &input_bytes));
    TF_LITE_ENSURE_OK(context, TfLiteEvalTensorByteLength(
                                   subgraph_input, &subgraph_input_bytes));
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, subgraph_input->type);
    TF_LITE_ENSURE_EQ(context, input_bytes, subgraph_input_bytes);

    void* source;
    void* destination;

    switch (copy_direction) {
      case CopyInputDirection::kFromIfToIntermediate:
        destination = intermediate_copy_buffer;
        source = input->data.raw;
        break;
      case CopyInputDirection::kFromIntermediateToSubgraph:
        destination = subgraph_input->data.raw;
        source = intermediate_copy_buffer;
        break;
      case CopyInputDirection::kFromIntermediateToIf:
        destination = input->data.raw;
        source = intermediate_copy_buffer;
        break;
        // NO default. enum is owned by this file and new enum shall be handled.
    }
    memcpy(destination, source, input_bytes);
    intermediate_copy_buffer += input_bytes;
  }
  return kTfLiteOk;
}

TfLiteStatus CopyFromIfInputToIntermediateInput(
    TfLiteContext* context, TfLiteNode* node, MicroGraph& graph_info,
    int active_branch_subgraph_index) {
  return CopyInputBetweenIfAndSubgraph(
      context, node, graph_info, active_branch_subgraph_index,
      CopyInputDirection::kFromIfToIntermediate);
}

TfLiteStatus CopyFromIntermediateInputToIfInput(
    TfLiteContext* context, TfLiteNode* node, MicroGraph& graph_info,
    int active_branch_subgraph_index) {
  return CopyInputBetweenIfAndSubgraph(
      context, node, graph_info, active_branch_subgraph_index,
      CopyInputDirection::kFromIntermediateToIf);
}
TfLiteStatus CopyFromIntermediateInputToSubgraphInput(
    TfLiteContext* context, TfLiteNode* node, MicroGraph& graph_info,
    int active_branch_subgraph_index) {
  return CopyInputBetweenIfAndSubgraph(
      context, node, graph_info, active_branch_subgraph_index,
      CopyInputDirection::kFromIntermediateToSubgraph);
}

TfLiteStatus CopyFromIntermediateOutputToIfOutput(
    TfLiteContext* context, TfLiteNode* node, MicroGraph& graph_info,
    int active_branch_subgraph_index) {
  return CopyOutputBetweenIfAndSubgraph(
      context, node, graph_info, active_branch_subgraph_index,
      CopyOutputDirection::kFromIntermediateToIf);
}

TfLiteStatus CopyFromSubgraphOutputToIntermediateOutput(
    TfLiteContext* context, TfLiteNode* node, MicroGraph& graph_info,
    int active_branch_subgraph_index) {
  return CopyOutputBetweenIfAndSubgraph(
      context, node, graph_info, active_branch_subgraph_index,
      CopyOutputDirection::kFromSubgraphToIntermediate);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
  TfLiteTensor* cond = micro_context->AllocateTempInputTensor(node, 0);

  TF_LITE_ENSURE(context, cond != nullptr);
  bool cond_value = cond->data.b[0];
  micro_context->DeallocateTempTfLiteTensor(cond);

  MicroGraph& graph_info = micro_context->graph();
  // Currently we copy the input / output between the subgraphs.
  int active_branch_subgraph_index =
      cond_value ? op_data->then_subgraph_index : op_data->else_subgraph_index;

  // The tensors of the IF op's subgraph and the active branch subgraphs are
  // planned separately and can overlap. So copying one tensor of the IF op to
  // one other tensor of the active subgraph can unintentionally overwrite
  // another tensor of the IF op. So we need to copy all input tensor of the IF
  // op to an intermediate buffer and then copied to input tensors of the active
  // branch subgraph.
  CopyFromIfInputToIntermediateInput(context, node, graph_info,
                                     active_branch_subgraph_index);
  CopyFromIntermediateInputToSubgraphInput(context, node, graph_info,
                                           active_branch_subgraph_index);

  TF_LITE_ENSURE_OK(context,
                    graph_info.InvokeSubgraph(active_branch_subgraph_index));

  // Copying to the IF op's tensors can unintentionally
  // overwrite the output tensor of the active subgraph. So we need to store all
  // output tensor of the active branch subgraph to an intermediate buffer and
  // then copy them to the IF op's output tensors.
  CopyFromSubgraphOutputToIntermediateOutput(context, node, graph_info,
                                             active_branch_subgraph_index);
  CopyFromIntermediateOutputToIfOutput(context, node, graph_info,
                                       active_branch_subgraph_index);

  // The IF op's input tensors may already get overwritten; these tensors may
  // still be used by other OPs and we need to restore them first from
  // intermediate buffers.
  CopyFromIntermediateInputToIfInput(context, node, graph_info,
                                     active_branch_subgraph_index);

  return kTfLiteOk;
}

}  // namespace.

TfLiteRegistration Register_IF() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
