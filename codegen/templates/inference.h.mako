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

#pragma once

#include "codegen/runtime/micro_codegen_context.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"

namespace ${model_name} {

class Model {
 public:
  Model();

  TfLiteStatus Invoke();

 private:
  TfLiteContext context_ = {};
  tflite::Subgraph subgraphs_[${len(graph.subgraphs)}];
  tflite::MicroCodegenContext micro_context_;
% for subgraph in graph.subgraphs:
  TfLiteNode ${subgraph.nodes_array}[${len(subgraph.operators)}] = {};
% endfor
% for subgraph in graph.subgraphs:
  TfLiteEvalTensor ${subgraph.tensors_array}[${len(subgraph.tensors)}] = {};
% endfor
};

}  // namespace ${model_name}
