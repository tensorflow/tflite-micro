/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {

constexpr int kInputVariableId = 0;
constexpr int kInputValue = 1;

#ifdef USE_TFLM_COMPRESSION

struct OpData {
  // scratch buffer for compressed input tensor
  int scratch_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

#endif  // USE_TFLM_COMPRESSION

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 0);

  // This must be a TfLiteEvalTensor despite this being in Prepare, because
  // CreateTensor allocates a temp tensor from the flatbuffer, which does not
  // contain the correct ID generated within the VAR_HANDLE op. EvalTensors are
  // all allocated during StartModelAllocation which happens before
  // init/prepare, and VAR_HANDLE Prepare() references its own op_data in the
  // TfLiteEvalTensor, so reading the ID here is valid.
  const TfLiteEvalTensor* input_resource_id_tensor =
      tflite::micro::GetEvalInput(context, node, kInputVariableId);
  TFLITE_DCHECK(input_resource_id_tensor != nullptr);
  TF_LITE_ENSURE(context, (input_resource_id_tensor->type == kTfLiteResource ||
                           input_resource_id_tensor->type == kTfLiteInt32));
  TF_LITE_ENSURE_EQ(context, NumElements(input_resource_id_tensor->dims), 1);

  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
  TfLiteTensor* input_value =
      micro_context->AllocateTempInputTensor(node, kInputValue);
  TFLITE_DCHECK(input_value != nullptr);

  MicroGraph& graph_info = micro_context->graph();

  MicroResourceVariables* resources = graph_info.GetResourceVariables();
  // If the data field of this tensor is nullptr, we assume that this is a case
  // of using resource variables in another subgraph, and the resource_id
  // will be valid during Eval time. In case it wasn't valid, this will
  // still be caught during Invoke. More info in b/277231654.
  if (input_resource_id_tensor->data.i32 != nullptr) {
    TF_LITE_ENSURE_OK(context,
                      resources->Allocate(input_resource_id_tensor->data.i32[0],
                                          context, input_value));
  }

#ifdef USE_TFLM_COMPRESSION

  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);
  // Compression scratch buffers.
  // These will only be allocated if the tensor is compressed.
  data->scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(node, kInputValue);

#endif  // USE_TFLM_COMPRESSION

  micro_context->DeallocateTempTfLiteTensor(input_value);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input_id =
      tflite::micro::GetEvalInput(context, node, kInputVariableId);
  TFLITE_DCHECK(input_id != nullptr);

  const TfLiteEvalTensor* input_value =
      tflite::micro::GetEvalInput(context, node, kInputValue);
  TFLITE_DCHECK(input_value != nullptr);

  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
  MicroGraph& graph_info = micro_context->graph();

  MicroResourceVariables* resources = graph_info.GetResourceVariables();
  if (resources == nullptr) {
    MicroPrintf(
        "ASSIGN_VARIABLE requires resource variables. Please create "
        "ResourceVariables and pass it to the interpreter.");
    return kTfLiteError;
  }

#ifdef USE_TFLM_COMPRESSION
  OpData* data = static_cast<OpData*>(node->user_data);
  const CompressionTensorData* comp_td =
      micro_context->GetTensorCompressionData(node, kInputValue);
  const void* buffer = tflite::micro::GetTensorData<void>(
      micro_context, input_value, comp_td, data->scratch_index);
#else   // USE_TFLM_COMPRESSION
  const void* buffer = tflite::micro::GetTensorData<void>(input_value);
#endif  // USE_TFLM_COMPRESSION

  TF_LITE_ENSURE_OK(context,
                    resources->Assign(input_id->data.i32[0],
                                      EvalTensorBytes(input_value), buffer));
  return kTfLiteOk;
}

}  // namespace.

#ifdef USE_TFLM_COMPRESSION

TFLMRegistration Register_ASSIGN_VARIABLE() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);

#else  // USE_TFLM_COMPRESSION

TFLMRegistration Register_ASSIGN_VARIABLE() {
  return tflite::micro::RegisterOp(nullptr, Prepare, Eval);

#endif  // USE_TFLM_COMPRESSION
}

}  // namespace tflite
