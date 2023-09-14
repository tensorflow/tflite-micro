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

#include "codegen/runtime/micro_codegen_context.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace hello_world_model {
namespace {
// TODO(rjascani): We should probably split out the OpTable to a separate file
// once we start generating for multiple models.
enum OpCode { kFullyConnected, kCount };

TFLMInferenceRegistration op_table[OpCode::kCount] = {
    tflite::RegisterInference_FULLY_CONNECTED(),
};

// buffer_1 is located in the arena

alignas(16) uint8_t buffer_2[4] = {
    0xAD,
    0x01,
    0x00,
    0x00,
};

alignas(16) uint8_t buffer_3[16] = {
    0xD9, 0x3B, 0x27, 0x15, 0x1C, 0xE0, 0xDE, 0xDD,
    0x0F, 0x1B, 0xC5, 0xD7, 0x12, 0xDD, 0xF9, 0x7F,
};

alignas(16) uint8_t buffer_4[64] = {
    0x27, 0xFD, 0xFF, 0xFF, 0xA2, 0x07, 0x00, 0x00, 0x62, 0x02, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0xF1, 0x00, 0x00, 0x00, 0x29, 0xFE,
    0xFF, 0xFF, 0xDD, 0xFF, 0xFF, 0xFF, 0x9D, 0xFC, 0xFF, 0xFF, 0x3B,
    0x02, 0x00, 0x00, 0x45, 0x02, 0x00, 0x00, 0xA4, 0x10, 0x00, 0x00,
    0x67, 0x0F, 0x00, 0x00, 0x4F, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x87, 0xFC, 0xFF, 0xFF, 0x11, 0xEC, 0xFF, 0xFF,
};

alignas(16) uint8_t buffer_5[256] = {
    0xF4, 0x1A, 0xED, 0x09, 0x19, 0x21, 0xF4, 0x24, 0xE0, 0x21, 0xEF, 0xBC,
    0xF7, 0xF5, 0xFA, 0x19, 0x03, 0xDC, 0xD2, 0x02, 0x06, 0xF9, 0xF4, 0x02,
    0xFF, 0xFA, 0xEF, 0xF1, 0xEF, 0xD3, 0x27, 0xE1, 0xFB, 0x27, 0xDD, 0xEB,
    0xDB, 0xE4, 0x05, 0x1A, 0x17, 0xFC, 0x24, 0x12, 0x15, 0xEF, 0x1E, 0xE4,
    0x10, 0xFE, 0x14, 0xDA, 0x1C, 0xF8, 0xF3, 0xF1, 0xEF, 0xE2, 0xF3, 0x09,
    0xE3, 0xE9, 0xED, 0xE3, 0xE4, 0x15, 0x07, 0x0B, 0x04, 0x1B, 0x1A, 0xFE,
    0xEB, 0x01, 0xDE, 0x21, 0xE6, 0x0B, 0xEC, 0x03, 0x23, 0x0A, 0x22, 0x24,
    0x1E, 0x27, 0x03, 0xE6, 0x03, 0x24, 0xFF, 0xC0, 0x11, 0xF8, 0xFC, 0xF1,
    0x11, 0x0C, 0xF5, 0xE0, 0xF3, 0x07, 0x17, 0xE5, 0xE8, 0xED, 0xFA, 0xDC,
    0xE8, 0x23, 0xFB, 0x07, 0xDD, 0xFB, 0xFD, 0x00, 0x14, 0x26, 0x11, 0x17,
    0xE7, 0xF1, 0x11, 0xEA, 0x02, 0x26, 0x04, 0x04, 0x25, 0x21, 0x1D, 0x0A,
    0xDB, 0x1D, 0xDC, 0x20, 0x01, 0xFA, 0xE3, 0x37, 0x0B, 0xF1, 0x1A, 0x16,
    0xEF, 0x1C, 0xE7, 0x03, 0xE0, 0x16, 0x02, 0x03, 0x21, 0x18, 0x09, 0x2E,
    0xD9, 0xE5, 0x14, 0x0B, 0xEA, 0x1A, 0xFC, 0xD8, 0x13, 0x00, 0xC4, 0xD8,
    0xEC, 0xD9, 0xFE, 0x0D, 0x19, 0x20, 0xD8, 0xD6, 0xE2, 0x1F, 0xE9, 0xD7,
    0xCA, 0xE2, 0xDD, 0xC6, 0x13, 0xE7, 0x04, 0x3E, 0x00, 0x01, 0x14, 0xC7,
    0xDB, 0xE7, 0x15, 0x15, 0xF5, 0x06, 0xD6, 0x1A, 0xDC, 0x09, 0x22, 0xFE,
    0x08, 0x02, 0x13, 0xEF, 0x19, 0x1E, 0xE2, 0x09, 0xFD, 0xF3, 0x14, 0xDD,
    0xDA, 0x20, 0xD9, 0x0F, 0xE3, 0xF9, 0xF7, 0xEE, 0xE9, 0x24, 0xE6, 0x29,
    0x00, 0x07, 0x16, 0xE2, 0x1E, 0x0D, 0x23, 0xD3, 0xDD, 0xF7, 0x14, 0xFA,
    0x08, 0x22, 0x26, 0x21, 0x09, 0x08, 0x0F, 0x0B, 0xE0, 0x12, 0xF4, 0x7F,
    0xDC, 0x58, 0xE5, 0x26,
};

alignas(16) uint8_t buffer_6[64] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC2, 0xEA, 0xFF,
    0xFF, 0x75, 0xEA, 0xFF, 0xFF, 0xB8, 0xFA, 0xFF, 0xFF, 0x24, 0xFA,
    0xFF, 0xFF, 0xC8, 0xEF, 0xFF, 0xFF, 0xAC, 0xFF, 0xFF, 0xFF, 0x44,
    0x0D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xBD, 0x07, 0x00, 0x00,
    0x33, 0xEA, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xCC, 0xE4, 0xFF,
    0xFF, 0x4F, 0x0D, 0x00, 0x00, 0xCF, 0xE3, 0xFF, 0xFF,
};

alignas(16) uint8_t buffer_7[16] = {
    0xF7, 0xCA, 0x39, 0x47, 0x68, 0x73, 0x62, 0x63,
    0x40, 0xE6, 0x7F, 0x19, 0xAE, 0x44, 0x5F, 0x56,
};

// buffer_8 is located in the arena

// buffer_9 is located in the arena

// buffer_10 is located in the arena

constexpr size_t kSubgraph0Inputs[1] = {0};

constexpr size_t kSubgraph0Outputs[1] = {9};

struct Node0_0 {
  struct Inputs {
    int size = 3;
    int data[3] = {0, 6, 5};
  } inputs;
  struct Outputs {
    int size = 1;
    int data[1] = {7};
  } outputs;
  // No intermediates
  TfLiteFullyConnectedParams builtin_data = {
      .activation = kTfLiteActRelu,
      .weights_format = kTfLiteFullyConnectedWeightsFormatDefault,
      .keep_num_dims = false,
      .asymmetric_quantize_inputs = false,
      .quantized_bias_type = kTfLiteNoType};
} node_0_0;

struct Node0_1 {
  struct Inputs {
    int size = 3;
    int data[3] = {7, 4, 3};
  } inputs;
  struct Outputs {
    int size = 1;
    int data[1] = {8};
  } outputs;
  // No intermediates
  TfLiteFullyConnectedParams builtin_data = {
      .activation = kTfLiteActRelu,
      .weights_format = kTfLiteFullyConnectedWeightsFormatDefault,
      .keep_num_dims = false,
      .asymmetric_quantize_inputs = false,
      .quantized_bias_type = kTfLiteNoType};
} node_0_1;

struct Node0_2 {
  struct Inputs {
    int size = 3;
    int data[3] = {8, 2, 1};
  } inputs;
  struct Outputs {
    int size = 1;
    int data[1] = {9};
  } outputs;
  // No intermediates
  TfLiteFullyConnectedParams builtin_data = {
      .activation = kTfLiteActNone,
      .weights_format = kTfLiteFullyConnectedWeightsFormatDefault,
      .keep_num_dims = false,
      .asymmetric_quantize_inputs = false,
      .quantized_bias_type = kTfLiteNoType};
} node_0_2;

struct Tensor0_0Dims {
  int size = 2;
  int data[2] = {1, 1};
} tensor0_0_dims;

struct Tensor0_1Dims {
  int size = 1;
  int data[1] = {1};
} tensor0_1_dims;

struct Tensor0_2Dims {
  int size = 2;
  int data[2] = {1, 16};
} tensor0_2_dims;

struct Tensor0_3Dims {
  int size = 1;
  int data[1] = {16};
} tensor0_3_dims;

struct Tensor0_4Dims {
  int size = 2;
  int data[2] = {16, 16};
} tensor0_4_dims;

struct Tensor0_5Dims {
  int size = 1;
  int data[1] = {16};
} tensor0_5_dims;

struct Tensor0_6Dims {
  int size = 2;
  int data[2] = {16, 1};
} tensor0_6_dims;

struct Tensor0_7Dims {
  int size = 2;
  int data[2] = {1, 16};
} tensor0_7_dims;

struct Tensor0_8Dims {
  int size = 2;
  int data[2] = {1, 16};
} tensor0_8_dims;

struct Tensor0_9Dims {
  int size = 2;
  int data[2] = {1, 1};
} tensor0_9_dims;

TfLiteStatus InvokeSubgraph0(TfLiteContext* context,
                             tflite::Span<TfLiteNode> nodes) {
  TFLITE_DCHECK(nodes.size() == 3);
  TF_LITE_ENSURE_OK(
      context, op_table[OpCode::kFullyConnected].invoke(context, &nodes[0]));
  TF_LITE_ENSURE_OK(
      context, op_table[OpCode::kFullyConnected].invoke(context, &nodes[1]));
  TF_LITE_ENSURE_OK(
      context, op_table[OpCode::kFullyConnected].invoke(context, &nodes[2]));

  return kTfLiteOk;
}

}  // namespace

Model::Model()
  : subgraphs_{
      {.inputs = {&kSubgraph0Inputs[0], 1},
       .outputs = {&kSubgraph0Outputs[0], 1},
       .nodes = {&subgraph0_nodes_[0], 3},
       .tensors = {&subgraph0_tensors_[0], 10},
       .invoke = &InvokeSubgraph0},
    },
    micro_context_{&context_, {&subgraphs_[0], 1}} {
  context_.impl_ = static_cast<void*>(&micro_context_);
  context_.ReportError = nullptr;
  context_.GetTensor = nullptr;
  context_.GetEvalTensor = tflite::MicroContextGetEvalTensor;
  context_.profiler = nullptr;
  context_.GetExternalContext = nullptr;
  context_.GetScratchBuffer = nullptr;

  subgraph0_nodes_[0] = TfLiteNode{
      .inputs = reinterpret_cast<TfLiteIntArray*>(&node_0_0.inputs),
      .outputs = reinterpret_cast<TfLiteIntArray*>(&node_0_0.outputs),
      .intermediates = nullptr,
      .user_data = nullptr,
      .builtin_data = static_cast<void*>(&node_0_0.builtin_data),
      .custom_initial_data = nullptr,
      .custom_initial_data_size = 0};
  subgraph0_nodes_[1] = TfLiteNode{
      .inputs = reinterpret_cast<TfLiteIntArray*>(&node_0_1.inputs),
      .outputs = reinterpret_cast<TfLiteIntArray*>(&node_0_1.outputs),
      .intermediates = nullptr,
      .user_data = nullptr,
      .builtin_data = static_cast<void*>(&node_0_1.builtin_data),
      .custom_initial_data = nullptr,
      .custom_initial_data_size = 0};
  subgraph0_nodes_[2] = TfLiteNode{
      .inputs = reinterpret_cast<TfLiteIntArray*>(&node_0_2.inputs),
      .outputs = reinterpret_cast<TfLiteIntArray*>(&node_0_2.outputs),
      .intermediates = nullptr,
      .user_data = nullptr,
      .builtin_data = static_cast<void*>(&node_0_2.builtin_data),
      .custom_initial_data = nullptr,
      .custom_initial_data_size = 0};

  subgraph0_tensors_[0] = TfLiteEvalTensor{
      .data = {.data = static_cast<void*>(nullptr /* buffer_1 */)},
      .dims = reinterpret_cast<TfLiteIntArray*>(&tensor0_0_dims),
      .type = kTfLiteInt8};
  subgraph0_tensors_[1] = TfLiteEvalTensor{
      .data = {.data = static_cast<void*>(&buffer_2)},
      .dims = reinterpret_cast<TfLiteIntArray*>(&tensor0_1_dims),
      .type = kTfLiteInt32};
  subgraph0_tensors_[2] = TfLiteEvalTensor{
      .data = {.data = static_cast<void*>(&buffer_3)},
      .dims = reinterpret_cast<TfLiteIntArray*>(&tensor0_2_dims),
      .type = kTfLiteInt8};
  subgraph0_tensors_[3] = TfLiteEvalTensor{
      .data = {.data = static_cast<void*>(&buffer_4)},
      .dims = reinterpret_cast<TfLiteIntArray*>(&tensor0_3_dims),
      .type = kTfLiteInt32};
  subgraph0_tensors_[4] = TfLiteEvalTensor{
      .data = {.data = static_cast<void*>(&buffer_5)},
      .dims = reinterpret_cast<TfLiteIntArray*>(&tensor0_4_dims),
      .type = kTfLiteInt8};
  subgraph0_tensors_[5] = TfLiteEvalTensor{
      .data = {.data = static_cast<void*>(&buffer_6)},
      .dims = reinterpret_cast<TfLiteIntArray*>(&tensor0_5_dims),
      .type = kTfLiteInt32};
  subgraph0_tensors_[6] = TfLiteEvalTensor{
      .data = {.data = static_cast<void*>(&buffer_7)},
      .dims = reinterpret_cast<TfLiteIntArray*>(&tensor0_6_dims),
      .type = kTfLiteInt8};
  subgraph0_tensors_[7] = TfLiteEvalTensor{
      .data = {.data = static_cast<void*>(nullptr /* buffer_8 */)},
      .dims = reinterpret_cast<TfLiteIntArray*>(&tensor0_7_dims),
      .type = kTfLiteInt8};
  subgraph0_tensors_[8] = TfLiteEvalTensor{
      .data = {.data = static_cast<void*>(nullptr /* buffer_9 */)},
      .dims = reinterpret_cast<TfLiteIntArray*>(&tensor0_8_dims),
      .type = kTfLiteInt8};
  subgraph0_tensors_[9] = TfLiteEvalTensor{
      .data = {.data = static_cast<void*>(nullptr /* buffer_10 */)},
      .dims = reinterpret_cast<TfLiteIntArray*>(&tensor0_9_dims),
      .type = kTfLiteInt8};
}

TfLiteStatus Model::Invoke() { return micro_context_.InvokeSubgraph(0); }

}  // namespace hello_world_model
