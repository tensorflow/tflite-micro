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

#include "hello_world_model.h"

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace hello_world_model {
namespace {
// TODO(rjascani): We should probably split out the OpTable to a separate file
// once we start generating for multiple models.
enum OpCode { kFullyConnected, kCount };

TFLMInferenceRegistration op_table[OpCode::kCount] = {
    tflite::RegisterInference_FULLY_CONNECTED(),
};

TfLiteIntArray node_0_0_inputs = {.size = 3, .data = {0, 6, 5}};
TfLiteIntArray node_0_0_outputs = {.size = 1, .data = {7}};

TfLiteIntArray node_0_1_inputs = {.size = 3, .data = {7, 4, 3}};
TfLiteIntArray node_0_1_outputs = {.size = 1, .data = {8}};

TfLiteIntArray node_0_2_inputs = {.size = 3, .data = {8, 2, 1}};
TfLiteIntArray node_0_2_outputs = {.size = 1, .data = {9}};

}  // namespace

Model::Model() {
  context_.impl_ = nullptr;
  context_.ReportError = nullptr;
  context_.GetTensor = nullptr;
  context_.GetEvalTensor = nullptr;
  context_.profiler = nullptr;
  context_.GetExternalContext = nullptr;
  context_.GetScratchBuffer = nullptr;

  subgraph0_nodes_[0] = {.inputs = &node_0_0_inputs,
                         .outputs = &node_0_0_outputs,
                         .intermediates = nullptr,
                         .user_data = nullptr,            // from preprocessor
                         .builtin_data = nullptr,         // from flatbuffer
                         .custom_initial_data = nullptr,  // from flatbuffer
                         .custom_initial_data_size = 0};
  subgraph0_nodes_[1] = {.inputs = &node_0_1_inputs,
                         .outputs = &node_0_1_outputs,
                         .intermediates = nullptr,
                         .user_data = nullptr,            // from preprocessor
                         .builtin_data = nullptr,         // from flatbuffer
                         .custom_initial_data = nullptr,  // from flatbuffer
                         .custom_initial_data_size = 0};
  subgraph0_nodes_[2] = {.inputs = &node_0_2_inputs,
                         .outputs = &node_0_2_outputs,
                         .intermediates = nullptr,
                         .user_data = nullptr,            // from preprocessor
                         .builtin_data = nullptr,         // from flatbuffer
                         .custom_initial_data = nullptr,  // from flatbuffer
                         .custom_initial_data_size = 0};
}

TfLiteStatus Model::Invoke() { return InvokeSubgraph0(); }

TfLiteStatus Model::InvokeSubgraph0() {
  TF_LITE_ENSURE_OK(context_, op_table[OpCode::kFullyConnected].invoke(
                                  &context_, &subgraph0_nodes_[0]));
  TF_LITE_ENSURE_OK(context_, op_table[OpCode::kFullyConnected].invoke(
                                  &context_, &subgraph0_nodes_[1]));
  TF_LITE_ENSURE_OK(context_, op_table[OpCode::kFullyConnected].invoke(
                                  &context_, &subgraph0_nodes_[2]));
  return kTfLiteOk;
}

}  // namespace hello_world_model
