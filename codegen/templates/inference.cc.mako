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

#include "tensorflow/lite/c/builtin_op_data.h"
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
  ${op_code.enum_name},
% endfor
  kCount
};

TFLMInferenceRegistration op_table[OpCode::kCount] = {
% for op_code in op_code_table.op_codes:
    ${op_code.register_function}(),
% endfor
};

% for buffer in graph.buffers:
${buffer.generate_c_buffer_array("")}
% endfor
% for subgraph in graph.subgraphs:
${subgraph.generate_c_node_data("")}

${subgraph.generate_c_tensor_data("")}
% endfor

% if graph.needs_zero_length_int_array:
TfLiteIntArray zero_length_int_array = {};
% endif
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
${subgraph.generate_c_node_init("  ")}

${subgraph.generate_c_tensor_init("  ")}
% endfor
}

TfLiteStatus Model::Invoke() { return InvokeSubgraph0(); }

% for subgraph in graph.subgraphs:
TfLiteStatus Model::InvokeSubgraph${subgraph.index}() {
${subgraph.generate_c_invoke("  ")}
  return kTfLiteOk;
}
% endfor

}  // namespace ${model_name}
