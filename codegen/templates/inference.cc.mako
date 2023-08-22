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

/* AUTOMATICALLY GENERATED DO NOT MODIFY */

#include "${header_file}"

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace ${model_name} {
namespace {
// TODO(rjascani): We should probably split out the OpTable to a separate file
// once we start generating for multiple models.
enum OpCode {
% for op_code in op_code_table.op_codes:
  ${op_code.enum_name()},
% endfor
  kCount
};

TFLMInferenceRegistration op_table[OpCode::kCount] = {
% for op_code in op_code_table.op_codes:
    ${op_code.register_function()}(),
% endfor
};

% for subgraph in graph.subgraphs:
% for operator in subgraph.operators:
${operator.node_data_c_struct}

% endfor
% endfor
// TODO(rjascani): Move this to a common utility header.
template <typename T>
inline TfLiteIntArray* ToIntArray(const T* array) {
  return reinterpret_cast<TfLiteIntArray*>(const_cast<T*>(array));
}

}  // namespace

Model::Model() {
  context_.impl_ = nullptr;
  context_.ReportError = nullptr;
  context_.GetTensor = nullptr;
  context_.GetEvalTensor = nullptr;
  context_.profiler = nullptr;
  context_.GetExternalContext = nullptr;
  context_.GetScratchBuffer = nullptr;

% for subgraph in graph.subgraphs:
  % for operator in subgraph.operators:
  ${operator.node_element} = {
    .inputs = ToIntArray(&${operator.node_data_inputs}),
    .outputs = ToIntArray(&${operator.node_data_outputs}),
% if operator.node_data_intermediates:
    .intermediates = ToIntArray(&${operator.node_data_intermediates});
% else:
    .intermediates = nullptr,
% endif
    .user_data = nullptr,     // from preprocessor
    .builtin_data = nullptr,  // from flatbuffer
    .custom_initial_data = nullptr, // from flatbuffer
    .custom_initial_data_size = 0};
  % endfor
% endfor
}

TfLiteStatus Model::Invoke() { return InvokeSubgraph0(); }

% for subgraph in graph.subgraphs:
TfLiteStatus Model::InvokeSubgraph${subgraph.index}() {
  % for operator in subgraph.operators:
  TF_LITE_ENSURE_OK(context_, op_table[OpCode::${operator.op_code.enum_name()}].invoke(
                                  &context_, &${operator.node_element}));
  % endfor
  return kTfLiteOk;
}
% endfor

}  // namespace ${model_name}
