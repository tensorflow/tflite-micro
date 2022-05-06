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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)

constexpr int max_num_input_tensors = 24;
constexpr int output_tensor_index = max_num_input_tensors;
constexpr int intermediate_tensor_base = max_num_input_tensors + 1;

constexpr int n_batch_integer_no_peephole = 2;
constexpr int n_input_integer_no_peephole = 5;
constexpr int n_cell_integer_no_peephole = 4;
constexpr int n_output_integer_no_peephole = 3;
constexpr int sequence_length_integer_no_peephole = 3;

// Model inputs. sequence -batch - input
const float lstm_input_integer_no_peephole[sequence_length_integer_no_peephole *
                                           n_batch_integer_no_peephole *
                                           n_input_integer_no_peephole] = {
    0.7, 0.8, 0.1, 0.2, 0.3,  //
    0.8, 0.1, 0.2, 0.4, 0.5,  //
    0.2, 0.7, 0.7, 0.1, 0.7,  //
    0.3, 0.2, 0.9, 0.8, 0.1,  //
    0.7, 0.8, 0.1, 0.2, 0.3,  //
    0.3, 0.2, 0.9, 0.8, 0.1,  //
};

int8_t lstm_input_integer_no_peephole_quantized
    [sequence_length_integer_no_peephole * n_batch_integer_no_peephole *
     n_input_integer_no_peephole];

const float
    input_to_input_weights_integer_no_peephole[n_cell_integer_no_peephole *
                                               n_input_integer_no_peephole] = {
        0.5,  0.6, 0.7,  -0.8, -0.9, 0.1,  0.2,  0.3,  -0.4, 0.5,
        -0.8, 0.7, -0.6, 0.5,  -0.4, -0.5, -0.4, -0.3, -0.2, -0.1};

int8_t input_to_input_weights_integer_no_peephole_quantized
    [n_cell_integer_no_peephole * n_input_integer_no_peephole];

const float
    input_to_forget_weights_integer_no_peephole[n_cell_integer_no_peephole *
                                                n_input_integer_no_peephole] = {
        -0.6, -0.1, 0.3,  0.2,  0.9,  -0.5, -0.2, -0.4, 0.3,  -0.8,
        -0.4, 0.3,  -0.5, -0.4, -0.6, 0.3,  -0.4, -0.6, -0.5, -0.5};

int8_t input_to_forget_weights_integer_no_peephole_quantized
    [n_cell_integer_no_peephole * n_input_integer_no_peephole];

const float
    input_to_cell_weights_integer_no_peephole[n_cell_integer_no_peephole *
                                              n_input_integer_no_peephole] = {
        -0.4, -0.3, -0.2, -0.1, -0.5, 0.5, -0.2, -0.3, -0.2, -0.6,
        0.6,  -0.1, -0.4, -0.3, -0.7, 0.7, -0.9, -0.5, 0.8,  0.6};

int8_t input_to_cell_weights_integer_no_peephole_quantized
    [n_cell_integer_no_peephole * n_input_integer_no_peephole];

const float
    input_to_output_weights_integer_no_peephole[n_cell_integer_no_peephole *
                                                n_input_integer_no_peephole] = {
        -0.8, -0.4, -0.2, -0.9, -0.1, -0.7, 0.3, -0.3, -0.8, -0.2,
        0.6,  -0.2, 0.4,  -0.7, -0.3, -0.5, 0.1, 0.5,  -0.6, -0.4};

int8_t input_to_output_weights_integer_no_peephole_quantized
    [n_cell_integer_no_peephole * n_input_integer_no_peephole];

const float recurrent_to_input_weights_integer_no_peephole
    [n_cell_integer_no_peephole * n_output_integer_no_peephole] = {
        -0.2, -0.3, 0.4, 0.1, -0.5, 0.9, -0.2, -0.3, -0.7, 0.05, -0.2, -0.6};

int8_t recurrent_to_input_weights_integer_no_peephole_quantized
    [n_cell_integer_no_peephole * n_output_integer_no_peephole];

const float recurrent_to_forget_weights_integer_no_peephole
    [n_cell_integer_no_peephole * n_output_integer_no_peephole] = {
        -0.5, -0.3, -0.5, -0.2, 0.6, 0.4, 0.9, 0.3, -0.1, 0.2, 0.5, 0.2};

int8_t recurrent_to_forget_weights_integer_no_peephole_quantized
    [n_cell_integer_no_peephole * n_output_integer_no_peephole];

const float recurrent_to_cell_weights_integer_no_peephole
    [n_cell_integer_no_peephole * n_output_integer_no_peephole] = {
        -0.3, 0.2, 0.1, -0.3, 0.8, -0.08, -0.2, 0.3, 0.8, -0.6, -0.1, 0.2};

int8_t recurrent_to_cell_weights_integer_no_peephole_quantized
    [n_cell_integer_no_peephole * n_output_integer_no_peephole];

const float recurrent_to_output_weights_integer_no_peephole
    [n_cell_integer_no_peephole * n_output_integer_no_peephole] = {
        0.3, -0.1, 0.1, -0.2, -0.5, -0.7, -0.2, -0.6, -0.1, -0.4, -0.7, -0.2};

int8_t recurrent_to_output_weights_integer_no_peephole_quantized
    [n_cell_integer_no_peephole * n_output_integer_no_peephole];

const float input_gate_bias_integer_no_peephole[n_cell_integer_no_peephole] = {
    0.03, 0.15, 0.22, 0.38};

int32_t
    input_gate_bias_integer_no_peephole_quantized[n_cell_integer_no_peephole];

const float forget_gate_bias_integer_no_peephole[n_cell_integer_no_peephole] = {
    0.1, -0.3, -0.2, 0.1};

int32_t
    forget_gate_bias_integer_no_peephole_quantized[n_cell_integer_no_peephole];

const float cell_gate_bias_integer_no_peephole[n_cell_integer_no_peephole] = {
    -0.05, 0.72, 0.25, 0.08};

int32_t
    cell_gate_bias_integer_no_peephole_quantized[n_cell_integer_no_peephole];

const float output_gate_bias_integer_no_peephole[n_cell_integer_no_peephole] = {
    0.05, -0.01, 0.2, 0.1};

const float
    projection_weights_integer_no_peephole[n_output_integer_no_peephole *
                                           n_cell_integer_no_peephole] = {
        -0.1, 0.2, 0.01, -0.2, 0.1, 0.5, 0.3, 0.08, 0.07, 0.2, -0.4, 0.2};

int8_t projection_weights_integer_no_peephole_quantized
    [n_output_integer_no_peephole * n_cell_integer_no_peephole];

int32_t
    output_gate_bias_integer_no_peephole_quantized[n_cell_integer_no_peephole];

int16_t output_state_no_peephole[n_batch_integer_no_peephole *
                                 n_cell_integer_no_peephole];

int16_t cell_state_no_peephole[n_batch_integer_no_peephole *
                               n_cell_integer_no_peephole];

const float input_layer_norm_coefficients_integer_no_peephole
    [n_cell_integer_no_peephole] = {0.1, 0.2, 0.3, 0.5};

int16_t input_layer_norm_coefficients_integer_no_peephole_quantized
    [n_cell_integer_no_peephole];

const float forget_layer_norm_coefficients_integer_no_peephole
    [n_cell_integer_no_peephole] = {0.2, 0.2, 0.4, 0.3};

int16_t forget_layer_norm_coefficients_integer_no_peephole_quantized
    [n_cell_integer_no_peephole];

const float cell_layer_norm_coefficients_integer_no_peephole
    [n_cell_integer_no_peephole] = {0.7, 0.2, 0.3, 0.8};

int16_t cell_layer_norm_coefficients_integer_no_peephole_quantized
    [n_cell_integer_no_peephole];

const float output_layer_norm_coefficients_integer_no_peephole
    [n_cell_integer_no_peephole] = {0.6, 0.2, 0.2, 0.5};

int16_t output_layer_norm_coefficients_integer_no_peephole_quantized
    [n_cell_integer_no_peephole];

// The scale and zero point of intermediate tensors.
float intermediate_scale_integer_no_peephole[5][2] = {
    {1.0f, 0.007059f}, {1.0f, 0.007812f}, {1.0f, 0.007059f},
    {1.0f, 0.007812f}, {1.0f, 0.007f},
};

int intermediate_zp_integer_no_peephole[5][2] = {
    {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0},
};

TfLiteAffineQuantization intermediate_qparam_integer_no_peephole[5];

int8_t output_no_peephole[n_batch_integer_no_peephole *
                          sequence_length_integer_no_peephole *
                          n_output_integer_no_peephole];

// Input ranges.
const float ranges_integer_no_peephole[25][2] = {
    {-1.0, 127.0 / 128},  // input tensor
    {-1.0, 1.0},          // input_to_input_weight tensor
    {-1.0, 1.0},          // input_to_forget_weight tensor
    {-1.0, 1.0},          // input_to_cell_weight tensor
    {-1.0, 1.0},          // input_to_output_weight tensor

    {-1.0, 1.0},  // recurrent_to_input_weight tensor
    {-1.0, 1.0},  // recurrent_to_forget_weight tensor
    {-1.0, 1.0},  // recurrent_to_cell_weight tensor
    {-1.0, 1.0},  // recurrent_to_output_weight tensor

    {-1, 1},  // cell_to_input_weight tensor
    {-1, 1},  // cell_to_forget_weight tensor
    {-1, 1},  // cell_to_output_weight tensor

    {-100, 100},  // input_gate_bias tensor
    {-100, 100},  // forget_gate_bias tensor
    {-100, 100},  // cell_gate_bias tensor
    {-100, 100},  // output_gate_bias tensor

    {-0.5, 0.5},  // projection_weight tensor
    {-1, 1},      // projection_bias tensor

    {-1.0, 32767.0 / 32768},  // output_state tensor
    {-1, 1},                  // cell_state tensor

    {-1.00001, 1.0},  // input_layer_norm_coefficient tensor
    {-1.00001, 1.0},  // forget_layer_norm_coefficient tensor
    {-1.00001, 1.0},  // cell_layer_norm_coefficient tensor
    {-1.00001, 1.0},  // output_layer_norm_coefficient tensor
    // Output scale is the same as output_state scale and only output_state
    // scale is used in the op, so this is only provided for clarity.
    {-1.0, 32767.0 / 32768},  // output tensor.
};

// Expected outputs, n_batch * sequence_length * n_output
const int8_t
    expected_output_integer_no_peephole[n_batch_integer_no_peephole *
                                        sequence_length_integer_no_peephole *
                                        n_output_integer_no_peephole] = {
        127,  127, -108, -67, 127, 127, -128, 127, 127,
        -128, 127, 127,  127, 127, 127, -128, 127, 127,
};

constexpr int n_batch_integer_peephole = 2;
constexpr int n_input_integer_peephole = 5;
constexpr int n_cell_integer_peephole = 4;
constexpr int n_output_integer_peephole = 3;
constexpr int sequence_length_integer_peephole = 3;

// Model inputs. sequence -batch - input
const float lstm_input_integer_peephole[sequence_length_integer_peephole *
                                        n_batch_integer_peephole *
                                        n_input_integer_peephole] = {
    0.7, 0.8, 0.1, 0.2, 0.3,  //
    0.8, 0.1, 0.2, 0.4, 0.5,  //
    0.2, 0.7, 0.7, 0.1, 0.7,  //
    0.3, 0.2, 0.9, 0.8, 0.1,  //
    0.7, 0.8, 0.1, 0.2, 0.3,  //
    0.3, 0.2, 0.9, 0.8, 0.1,  //
};

int8_t lstm_input_integer_peephole_quantized[sequence_length_integer_peephole *
                                             n_batch_integer_peephole *
                                             n_input_integer_peephole];

// Model related weights.
const float input_to_input_weights_integer_peephole[n_cell_integer_peephole *
                                                    n_input_integer_peephole] =
    {0.5,  0.6, 0.7,  -0.8, -0.9, 0.1,  0.2,  0.3,  -0.4, 0.5,
     -0.8, 0.7, -0.6, 0.5,  -0.4, -0.5, -0.4, -0.3, -0.2, -0.1};

int8_t
    input_to_input_weights_integer_peephole_quantized[n_cell_integer_peephole *
                                                      n_input_integer_peephole];

const float input_to_forget_weights_integer_peephole[n_cell_integer_peephole *
                                                     n_input_integer_peephole] =
    {-0.6, -0.1, 0.3,  0.2,  0.9,  -0.5, -0.2, -0.4, 0.3,  -0.8,
     -0.4, 0.3,  -0.5, -0.4, -0.6, 0.3,  -0.4, -0.6, -0.5, -0.5};

int8_t input_to_forget_weights_integer_peephole_quantized
    [n_cell_integer_peephole * n_input_integer_peephole];

const float input_to_cell_weights_integer_peephole[n_cell_integer_peephole *
                                                   n_input_integer_peephole] = {
    -0.4, -0.3, -0.2, -0.1, -0.5, 0.5, -0.2, -0.3, -0.2, -0.6,
    0.6,  -0.1, -0.4, -0.3, -0.7, 0.7, -0.9, -0.5, 0.8,  0.6};

int8_t
    input_to_cell_weights_integer_peephole_quantized[n_cell_integer_peephole *
                                                     n_input_integer_peephole];

const float input_to_output_weights_integer_peephole[n_cell_integer_peephole *
                                                     n_input_integer_peephole] =
    {-0.8, -0.4, -0.2, -0.9, -0.1, -0.7, 0.3, -0.3, -0.8, -0.2,
     0.6,  -0.2, 0.4,  -0.7, -0.3, -0.5, 0.1, 0.5,  -0.6, -0.4};

int8_t input_to_output_weights_integer_peephole_quantized
    [n_cell_integer_peephole * n_input_integer_peephole];

const float
    recurrent_to_input_weights_integer_peephole[n_cell_integer_peephole *
                                                n_output_integer_peephole] = {
        -0.2, -0.3, 0.4, 0.1, -0.5, 0.9, -0.2, -0.3, -0.7, 0.05, -0.2, -0.6};

int8_t recurrent_to_input_weights_integer_peephole_quantized
    [n_cell_integer_peephole * n_output_integer_peephole];

const float
    recurrent_to_forget_weights_integer_peephole[n_cell_integer_peephole *
                                                 n_output_integer_peephole] = {
        -0.5, -0.3, -0.5, -0.2, 0.6, 0.4, 0.9, 0.3, -0.1, 0.2, 0.5, 0.2};

int8_t recurrent_to_forget_weights_integer_peephole_quantized
    [n_cell_integer_peephole * n_output_integer_peephole];

const float
    recurrent_to_cell_weights_integer_peephole[n_cell_integer_peephole *
                                               n_output_integer_peephole] = {
        -0.3, 0.2, 0.1, -0.3, 0.8, -0.08, -0.2, 0.3, 0.8, -0.6, -0.1, 0.2};

int8_t recurrent_to_cell_weights_integer_peephole_quantized
    [n_cell_integer_peephole * n_output_integer_peephole];

const float
    recurrent_to_output_weights_integer_peephole[n_cell_integer_peephole *
                                                 n_output_integer_peephole] = {
        0.3, -0.1, 0.1, -0.2, -0.5, -0.7, -0.2, -0.6, -0.1, -0.4, -0.7, -0.2};

int8_t recurrent_to_output_weights_integer_peephole_quantized
    [n_cell_integer_peephole * n_output_integer_peephole];

const float cell_to_input_weights_integer_peephole[n_cell_integer_peephole] = {
    0.3, -0.1, 0.1, -0.2};

int16_t
    cell_to_input_weights_integer_peephole_quantized[n_cell_integer_peephole];

const float cell_to_forget_weights_integer_peephole[n_cell_integer_peephole] = {
    0.2, -0.1, 0.1, -0.2};

int16_t
    cell_to_forget_weights_integer_peephole_quantized[n_cell_integer_peephole];

const float cell_to_output_weights_integer_peephole[n_cell_integer_peephole] = {
    0.3, -0.1, 0.1, -0.3};

int16_t
    cell_to_output_weights_integer_peephole_quantized[n_cell_integer_peephole];

const float input_gate_bias_integer_peephole[n_cell_integer_peephole] = {
    0.03, 0.15, 0.22, 0.38};

int32_t input_gate_bias_integer_peephole_quantized[n_cell_integer_peephole];

const float forget_gate_bias_integer_peephole[n_cell_integer_peephole] = {
    0.1, -0.3, -0.2, 0.1};

int32_t forget_gate_bias_integer_peephole_quantized[n_cell_integer_peephole];

const float cell_gate_bias_integer_peephole[n_cell_integer_peephole] = {
    -0.05, 0.72, 0.25, 0.08};

int32_t cell_gate_bias_integer_peephole_quantized[n_cell_integer_peephole];

const float output_gate_bias_integer_peephole[n_cell_integer_peephole] = {
    0.05, -0.01, 0.2, 0.1};

int32_t output_gate_bias_integer_peephole_quantized[n_cell_integer_peephole];

const float projection_weights_integer_peephole[n_output_integer_peephole *
                                                n_cell_integer_peephole] = {
    -0.1, 0.2, 0.01, -0.2, 0.1, 0.5, 0.3, 0.08, 0.07, 0.2, -0.4, 0.2};

int8_t projection_weights_integer_peephole_quantized[n_output_integer_peephole *
                                                     n_cell_integer_peephole];

int16_t
    output_state_peephole[n_batch_integer_peephole * n_cell_integer_peephole];

int16_t cell_state_peephole[n_batch_integer_peephole * n_cell_integer_peephole];

const float
    input_layer_norm_coefficients_integer_peephole[n_cell_integer_peephole] = {
        0.1, 0.2, 0.3, 0.5};

int16_t input_layer_norm_coefficients_integer_peephole_quantized
    [n_cell_integer_peephole];

const float
    forget_layer_norm_coefficients_integer_peephole[n_cell_integer_peephole] = {
        0.2, 0.2, 0.4, 0.3};

int16_t forget_layer_norm_coefficients_integer_peephole_quantized
    [n_cell_integer_peephole];

const float
    cell_layer_norm_coefficients_integer_peephole[n_cell_integer_peephole] = {
        0.7, 0.2, 0.3, 0.8};

int16_t cell_layer_norm_coefficients_integer_peephole_quantized
    [n_cell_integer_peephole];

const float
    output_layer_norm_coefficients_integer_peephole[n_cell_integer_peephole] = {
        0.6, 0.2, 0.2, 0.5};
int16_t output_layer_norm_coefficients_integer_peephole_quantized
    [n_cell_integer_peephole];

int8_t output_peephole[n_batch_integer_peephole *
                       sequence_length_integer_peephole *
                       n_output_integer_peephole];

// Input ranges.
const float ranges_integer_peephole[25][2] = {
    {-1.0, 127.0 / 128},  // input tensor
    {-1.0, 1.0},          // input_to_input_weight tensor
    {-1.0, 1.0},          // input_to_forget_weight tensor
    {-1.0, 1.0},          // input_to_cell_weight tensor
    {-1.0, 1.0},          // input_to_output_weight tensor

    {-1.0, 1.0},  // recurrent_to_input_weight tensor
    {-0.9, 0.9},  // recurrent_to_forget_weight tensor
    {-1.0, 1.0},  // recurrent_to_cell_weight tensor
    {-1.0, 1.0},  // recurrent_to_output_weight tensor

    {-0.3, 0.3},  // cell_to_input_weight tensor
    {-0.3, 0.3},  // cell_to_forget_weight tensor
    {-0.3, 0.3},  // cell_to_output_weight tensor

    {-100, 100},  // input_gate_bias tensor
    {-100, 80},   // forget_gate_bias tensor
    {-100, 100},  // cell_gate_bias tensor
    {-100, 100},  // output_gate_bias tensor

    {-0.5, 0.5},  // projection_weight tensor
    {-1, 1},      // projection_bias tensor

    {-1.0, 32767.0 / 32768},  // output_state tensor
    {-1, 1},                  // cell_state tensor

    {-0.5, 0.5},  // input_layer_norm_coefficient tensor
    {-0.5, 0.5},  // forget_layer_norm_coefficient tensor
    {-1.0, 1.0},  // cell_layer_norm_coefficient tensor
    {-1.0, 1.0},  // output_layer_norm_coefficient tensor
    // Output scale is the same as output_state scale and only output_state
    // scale is used in the op, so this is only provided for clarity.
    {-1.0, 32767.0 / 32768},  // output tensor.
};

// The scale and zero point of intermediate tensors.
float intermediate_scale_integer_peephole[5][2] = {
    {1.0f, 0.007059f}, {1.0f, 0.007812f}, {1.0f, 0.007059f},
    {1.0f, 0.007812f}, {1.0f, 0.007f},
};

int intermediate_zp_integer_peephole[5][2] = {
    {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0},
};

TfLiteAffineQuantization intermediate_qparam_integer_peephole[5];

// Expected outputs, n_batch * sequence_length * n_output
const int8_t expected_output_integer_peephole[n_batch_integer_peephole *
                                              sequence_length_integer_peephole *
                                              n_output_integer_peephole] = {
    127,  127, -16, -21, 127, 127, 23,   127, 127,
    -128, 127, 127, 127, 127, 127, -128, 127, 127,
};

template <typename T>
QuantizationParams SetQuantizationParams(float f_min, float f_max) {
  QuantizationParams qparam;
  int32_t zero_point = 0;
  float scale = 0;
  const T qmin = std::numeric_limits<T>::min();
  const T qmax = std::numeric_limits<T>::max();
  const float qmin_double = qmin;
  const float qmax_double = qmax;
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  TFLITE_DCHECK_LE(f_min, 0);
  TFLITE_DCHECK_GE(f_max, 0);
  if (f_min == f_max) {
    // Special case where the min,max range is a point. Should be {0}.
    TFLITE_DCHECK_EQ(f_min, 0);
    TFLITE_DCHECK_EQ(f_max, 0);
    qparam.scale = static_cast<double>(scale);
    qparam.zero_point = zero_point;
    return qparam;
  }

  // General case.
  //
  // First determine the scale.
  scale = (f_max - f_min) / (qmax_double - qmin_double);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  const float zero_point_from_min = qmin_double - f_min / scale;
  const float zero_point_from_max = qmax_double - f_max / scale;

  const float zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(f_min / scale);

  const float zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(f_max / scale);

  const float zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  //  padding).

  T nudged_zero_point = 0;
  if (zero_point_double < qmin_double) {
    nudged_zero_point = qmin;
  } else if (zero_point_double > qmax_double) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<T>(round(zero_point_double));
  }

  // The zero point should always be in the range of quantized value,
  // // [qmin, qmax].
  TFLITE_DCHECK_GE(nudged_zero_point, qmin);
  TFLITE_DCHECK_LE(nudged_zero_point, qmax);

  zero_point = nudged_zero_point;
  // finally, return the values
  qparam.scale = static_cast<double>(scale);
  qparam.zero_point = zero_point;
  return qparam;
}

void TestUnidirectionalSequenceLstmInteger(
    int n_batch, int n_input, int n_cell, int n_output, int sequence_length,
    bool time_major, bool use_cifg, bool use_peephole,
    bool use_projection_weights, bool use_projection_bias, bool use_layer_norm,
    bool use_8x8_8_implementation, const float ranges[][2],
    float intermediate_scale[5][2], int intermediate_zp[5][2],
    TfLiteAffineQuantization intermediate_qparam[5],

    const float* input, int8_t* input_quantized,

    const float* input_to_input_weights,
    int8_t* input_to_input_weights_quantized,
    const float* input_to_forget_weights,
    int8_t* input_to_forget_weights_quantized,
    const float* input_to_cell_weights, int8_t* input_to_cell_weights_quantized,
    const float* input_to_output_weights,
    int8_t* input_to_output_weights_quantized,

    const float* recurrent_to_input_weights,
    int8_t* recurrent_to_input_weights_quantized,
    const float* recurrent_to_forget_weights,
    int8_t* recurrent_to_forget_weights_quantized,
    const float* recurrent_to_cell_weights,
    int8_t* recurrent_to_cell_weights_quantized,
    const float* recurrent_to_output_weights,
    int8_t* recurrent_to_output_weights_quantized,

    const float* cell_to_input_weights,
    int16_t* cell_to_input_weights_quantized,
    const float* cell_to_forget_weights,
    int16_t* cell_to_forget_weights_quantized,
    const float* cell_to_output_weights,
    int16_t* cell_to_output_weights_quantized,

    const float* input_gate_bias, int32_t* input_gate_bias_quantized,
    const float* forget_gate_bias, int32_t* forget_gate_bias_quantized,
    const float* cell_gate_bias, int32_t* cell_gate_bias_quantized,
    const float* output_gate_bias, int32_t* output_gate_bias_quantized,

    const float* projection_weights, int8_t* projection_weights_quantized,
    const float* projection_bias, int32_t* projection_bias_quantized,

    int16_t* output_state, int16_t* cell_state,

    const float* input_layer_norm_coefficients,
    int16_t* input_layer_norm_coefficients_quantized,
    const float* forget_layer_norm_coefficients,
    int16_t* forget_layer_norm_coefficients_quantized,
    const float* cell_layer_norm_coefficients,
    int16_t* cell_layer_norm_coefficients_quantized,
    const float* output_layer_norm_coefficients,
    int16_t* output_layer_norm_coefficients_quantized,

    int8_t* output, const int8_t* expected_output,

    bool asymmetric_quantize_inputs = false) {
  int inputs_array_data[25];
  int outputs_array_data[2] = {1, output_tensor_index};
  int intermediate_array_data[6] = {5,
                                    intermediate_tensor_base,
                                    intermediate_tensor_base + 1,
                                    intermediate_tensor_base + 2,
                                    intermediate_tensor_base + 3,
                                    intermediate_tensor_base + 4};

  if (use_layer_norm) {
    inputs_array_data[0] = 24;
  } else {
    inputs_array_data[0] = 20;
  }

  QuantizationParams quantization_params;

  TfLiteTensor tensors[max_num_input_tensors + 1 + 5];

  quantization_params = SetQuantizationParams<int8_t>(
      ranges[kLstmInputTensor][0], ranges[kLstmInputTensor][1]);
  int input_dim[4] = {3, sequence_length, n_batch, n_input};
  tensors[kLstmInputTensor] = CreateQuantizedTensor<int8_t>(
      input, input_quantized, IntArrayFromInts(input_dim),
      quantization_params.scale, quantization_params.zero_point);
  inputs_array_data[kLstmInputTensor + 1] = kLstmInputTensor;

  int input_weights_dim[3] = {2, n_cell, n_input};

  if (use_cifg) {
    inputs_array_data[kLstmInputToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  } else {
    quantization_params = SetQuantizationParams<int8_t>(
        ranges[kLstmInputToInputWeightsTensor][0],
        ranges[kLstmInputToInputWeightsTensor][1]);
    tensors[kLstmInputToInputWeightsTensor] = CreateQuantizedTensor<int8_t>(
        input_to_input_weights, input_to_input_weights_quantized,
        IntArrayFromInts(input_weights_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmInputToInputWeightsTensor + 1] =
        kLstmInputToInputWeightsTensor;
  }

  quantization_params =
      SetQuantizationParams<int8_t>(ranges[kLstmInputToForgetWeightsTensor][0],
                                    ranges[kLstmInputToForgetWeightsTensor][1]);
  tensors[kLstmInputToForgetWeightsTensor] = CreateQuantizedTensor<int8_t>(
      input_to_forget_weights, input_to_forget_weights_quantized,
      IntArrayFromInts(input_weights_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmInputToForgetWeightsTensor + 1] =
      kLstmInputToForgetWeightsTensor;

  quantization_params =
      SetQuantizationParams<int8_t>(ranges[kLstmInputToCellWeightsTensor][0],
                                    ranges[kLstmInputToCellWeightsTensor][1]);
  tensors[kLstmInputToCellWeightsTensor] = CreateQuantizedTensor<int8_t>(
      input_to_cell_weights, input_to_cell_weights_quantized,
      IntArrayFromInts(input_weights_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmInputToCellWeightsTensor + 1] =
      kLstmInputToCellWeightsTensor;

  quantization_params =
      SetQuantizationParams<int8_t>(ranges[kLstmInputToOutputWeightsTensor][0],
                                    ranges[kLstmInputToOutputWeightsTensor][1]);
  tensors[kLstmInputToOutputWeightsTensor] = CreateQuantizedTensor<int8_t>(
      input_to_output_weights, input_to_output_weights_quantized,
      IntArrayFromInts(input_weights_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmInputToOutputWeightsTensor + 1] =
      kLstmInputToOutputWeightsTensor;

  int recurrent_weights_dim[3] = {2, n_cell, n_output};
  if (use_cifg) {
    inputs_array_data[kLstmRecurrentToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  } else {
    quantization_params = SetQuantizationParams<int8_t>(
        ranges[kLstmRecurrentToInputWeightsTensor][0],
        ranges[kLstmRecurrentToInputWeightsTensor][1]);
    tensors[kLstmRecurrentToInputWeightsTensor] = CreateQuantizedTensor<int8_t>(
        recurrent_to_input_weights, recurrent_to_input_weights_quantized,
        IntArrayFromInts(recurrent_weights_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmRecurrentToInputWeightsTensor + 1] =
        kLstmRecurrentToInputWeightsTensor;
  }

  quantization_params = SetQuantizationParams<int8_t>(
      ranges[kLstmRecurrentToForgetWeightsTensor][0],
      ranges[kLstmRecurrentToForgetWeightsTensor][1]);
  tensors[kLstmRecurrentToForgetWeightsTensor] = CreateQuantizedTensor<int8_t>(
      recurrent_to_forget_weights, recurrent_to_forget_weights_quantized,
      IntArrayFromInts(recurrent_weights_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmRecurrentToForgetWeightsTensor + 1] =
      kLstmRecurrentToForgetWeightsTensor;

  quantization_params = SetQuantizationParams<int8_t>(
      ranges[kLstmRecurrentToCellWeightsTensor][0],
      ranges[kLstmRecurrentToCellWeightsTensor][1]);
  tensors[kLstmRecurrentToCellWeightsTensor] = CreateQuantizedTensor<int8_t>(
      recurrent_to_cell_weights, recurrent_to_cell_weights_quantized,
      IntArrayFromInts(recurrent_weights_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmRecurrentToCellWeightsTensor + 1] =
      kLstmRecurrentToCellWeightsTensor;

  quantization_params = SetQuantizationParams<int8_t>(
      ranges[kLstmRecurrentToOutputWeightsTensor][0],
      ranges[kLstmRecurrentToOutputWeightsTensor][1]);
  tensors[kLstmRecurrentToOutputWeightsTensor] = CreateQuantizedTensor<int8_t>(
      recurrent_to_output_weights, recurrent_to_output_weights_quantized,
      IntArrayFromInts(recurrent_weights_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmRecurrentToOutputWeightsTensor + 1] =
      kLstmRecurrentToOutputWeightsTensor;

  int cell_weights_dim[2] = {1, n_cell};
  if (use_peephole) {
    if (use_cifg) {
      inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
          kTfLiteOptionalTensor;
    } else {
      quantization_params = SetQuantizationParams<int16_t>(
          ranges[kLstmCellToInputWeightsTensor][0],
          ranges[kLstmCellToInputWeightsTensor][1]);
      tensors[kLstmCellToInputWeightsTensor] = CreateQuantizedTensor<int16_t>(
          cell_to_input_weights, cell_to_input_weights_quantized,
          IntArrayFromInts(cell_weights_dim), quantization_params.scale,
          quantization_params.zero_point);
      inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
          kLstmCellToInputWeightsTensor;
    }

    quantization_params = SetQuantizationParams<int16_t>(
        ranges[kLstmCellToForgetWeightsTensor][0],
        ranges[kLstmCellToForgetWeightsTensor][1]);
    tensors[kLstmCellToForgetWeightsTensor] = CreateQuantizedTensor<int16_t>(
        cell_to_forget_weights, cell_to_forget_weights_quantized,
        IntArrayFromInts(cell_weights_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmCellToForgetWeightsTensor + 1] =
        kLstmCellToForgetWeightsTensor;

    quantization_params = SetQuantizationParams<int16_t>(
        ranges[kLstmCellToOutputWeightsTensor][0],
        ranges[kLstmCellToOutputWeightsTensor][1]);
    tensors[kLstmCellToOutputWeightsTensor] = CreateQuantizedTensor<int16_t>(
        cell_to_output_weights, cell_to_output_weights_quantized,
        IntArrayFromInts(cell_weights_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmCellToOutputWeightsTensor + 1] =
        kLstmCellToOutputWeightsTensor;
  } else {
    inputs_array_data[kLstmCellToInputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellToForgetWeightsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellToOutputWeightsTensor + 1] =
        kTfLiteOptionalTensor;
  }

  int gate_bias_dim[2] = {1, n_cell};
  if (use_cifg) {
    inputs_array_data[kLstmInputGateBiasTensor + 1] = kTfLiteOptionalTensor;
  } else {
    quantization_params =
        SetQuantizationParams<int32_t>(ranges[kLstmInputGateBiasTensor][0],
                                       ranges[kLstmInputGateBiasTensor][1]);
    tensors[kLstmInputGateBiasTensor] = CreateQuantizedTensor<int32_t>(
        input_gate_bias, input_gate_bias_quantized,
        IntArrayFromInts(gate_bias_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmInputGateBiasTensor + 1] = kLstmInputGateBiasTensor;
  }

  quantization_params =
      SetQuantizationParams<int32_t>(ranges[kLstmForgetGateBiasTensor][0],
                                     ranges[kLstmForgetGateBiasTensor][1]);
  tensors[kLstmForgetGateBiasTensor] = CreateQuantizedTensor<int32_t>(
      forget_gate_bias, forget_gate_bias_quantized,
      IntArrayFromInts(gate_bias_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmForgetGateBiasTensor + 1] = kLstmForgetGateBiasTensor;

  quantization_params = SetQuantizationParams<int32_t>(
      ranges[kLstmCellGateBiasTensor][0], ranges[kLstmCellGateBiasTensor][1]);
  tensors[kLstmCellGateBiasTensor] = CreateQuantizedTensor<int32_t>(
      cell_gate_bias, cell_gate_bias_quantized, IntArrayFromInts(gate_bias_dim),
      quantization_params.scale, quantization_params.zero_point);
  inputs_array_data[kLstmCellGateBiasTensor + 1] = kLstmCellGateBiasTensor;

  quantization_params =
      SetQuantizationParams<int32_t>(ranges[kLstmOutputGateBiasTensor][0],
                                     ranges[kLstmOutputGateBiasTensor][1]);
  tensors[kLstmOutputGateBiasTensor] = CreateQuantizedTensor<int32_t>(
      output_gate_bias, output_gate_bias_quantized,
      IntArrayFromInts(gate_bias_dim), quantization_params.scale,
      quantization_params.zero_point);
  inputs_array_data[kLstmOutputGateBiasTensor + 1] = kLstmOutputGateBiasTensor;

  int projection_weights_dim[3] = {2, n_output, n_cell};
  if (use_projection_weights) {
    quantization_params =
        SetQuantizationParams<int8_t>(ranges[kLstmProjectionWeightsTensor][0],
                                      ranges[kLstmProjectionWeightsTensor][1]);
    tensors[kLstmProjectionWeightsTensor] = CreateQuantizedTensor<int8_t>(
        projection_weights, projection_weights_quantized,
        IntArrayFromInts(projection_weights_dim), quantization_params.scale,
        quantization_params.zero_point);
    inputs_array_data[kLstmProjectionWeightsTensor + 1] =
        kLstmProjectionWeightsTensor;

    int projection_bias_dim[2] = {1, n_output};
    if (use_projection_bias) {
      quantization_params =
          SetQuantizationParams<int32_t>(ranges[kLstmProjectionBiasTensor][0],
                                         ranges[kLstmProjectionBiasTensor][1]);
      tensors[kLstmProjectionBiasTensor] = CreateQuantizedTensor<int32_t>(
          projection_bias, projection_bias_quantized,
          IntArrayFromInts(projection_bias_dim), quantization_params.scale,
          quantization_params.zero_point);
      inputs_array_data[kLstmProjectionBiasTensor + 1] =
          kLstmProjectionBiasTensor;
    } else {
      inputs_array_data[kLstmProjectionBiasTensor + 1] = kTfLiteOptionalTensor;
    }
  } else {
    inputs_array_data[kLstmProjectionWeightsTensor + 1] = kTfLiteOptionalTensor;
    inputs_array_data[kLstmProjectionBiasTensor + 1] = kTfLiteOptionalTensor;
  }

  int output_state_dim[3] = {2, n_batch, n_output};
  quantization_params = SetQuantizationParams<int16_t>(
      ranges[kLstmOutputStateTensor][0], ranges[kLstmOutputStateTensor][1]);
  tensors[kLstmOutputStateTensor] = CreateQuantizedTensor<int16_t>(
      output_state, IntArrayFromInts(output_state_dim),
      quantization_params.scale, quantization_params.zero_point, true);
  inputs_array_data[kLstmOutputStateTensor + 1] = kLstmOutputStateTensor;

  int cell_state_dim[3] = {2, n_batch, n_cell};
  quantization_params = SetQuantizationParams<int16_t>(
      ranges[kLstmCellStateTensor][0], ranges[kLstmCellStateTensor][1]);
  tensors[kLstmCellStateTensor] = CreateQuantizedTensor<int16_t>(
      cell_state, IntArrayFromInts(cell_state_dim), quantization_params.scale,
      quantization_params.zero_point, true);
  inputs_array_data[kLstmCellStateTensor + 1] = kLstmCellStateTensor;

  int layer_norm_dim[2] = {1, n_cell};
  if (use_layer_norm) {
    if (use_cifg) {
      inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
          kTfLiteOptionalTensor;
    } else {
      quantization_params = SetQuantizationParams<int16_t>(
          ranges[kLstmInputLayerNormCoefficientsTensor][0],
          ranges[kLstmInputLayerNormCoefficientsTensor][1]);
      tensors[kLstmInputLayerNormCoefficientsTensor] =
          CreateQuantizedTensor<int16_t>(
              input_layer_norm_coefficients,
              input_layer_norm_coefficients_quantized,
              IntArrayFromInts(layer_norm_dim), quantization_params.scale,
              quantization_params.zero_point);
      inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
          kLstmInputLayerNormCoefficientsTensor;
    }

    quantization_params = SetQuantizationParams<int16_t>(
        ranges[kLstmForgetLayerNormCoefficientsTensor][0],
        ranges[kLstmForgetLayerNormCoefficientsTensor][1]);
    tensors[kLstmForgetLayerNormCoefficientsTensor] =
        CreateQuantizedTensor<int16_t>(forget_layer_norm_coefficients,
                                       forget_layer_norm_coefficients_quantized,
                                       IntArrayFromInts(layer_norm_dim),
                                       quantization_params.scale,
                                       quantization_params.zero_point);
    inputs_array_data[kLstmForgetLayerNormCoefficientsTensor + 1] =
        kLstmForgetLayerNormCoefficientsTensor;

    quantization_params = SetQuantizationParams<int16_t>(
        ranges[kLstmCellLayerNormCoefficientsTensor][0],
        ranges[kLstmCellLayerNormCoefficientsTensor][1]);
    tensors[kLstmCellLayerNormCoefficientsTensor] =
        CreateQuantizedTensor<int16_t>(cell_layer_norm_coefficients,
                                       cell_layer_norm_coefficients_quantized,
                                       IntArrayFromInts(layer_norm_dim),
                                       quantization_params.scale,
                                       quantization_params.zero_point);
    inputs_array_data[kLstmCellLayerNormCoefficientsTensor + 1] =
        kLstmCellLayerNormCoefficientsTensor;

    quantization_params = SetQuantizationParams<int16_t>(
        ranges[kLstmOutputLayerNormCoefficientsTensor][0],
        ranges[kLstmOutputLayerNormCoefficientsTensor][1]);
    tensors[kLstmOutputLayerNormCoefficientsTensor] =
        CreateQuantizedTensor<int16_t>(output_layer_norm_coefficients,
                                       output_layer_norm_coefficients_quantized,
                                       IntArrayFromInts(layer_norm_dim),
                                       quantization_params.scale,
                                       quantization_params.zero_point);
    inputs_array_data[kLstmOutputLayerNormCoefficientsTensor + 1] =
        kLstmOutputLayerNormCoefficientsTensor;
  } else {
    inputs_array_data[kLstmInputLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmForgetLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmCellLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
    inputs_array_data[kLstmOutputLayerNormCoefficientsTensor + 1] =
        kTfLiteOptionalTensor;
  }
  int output_dim[4] = {3, sequence_length, n_batch, n_output};
  quantization_params = SetQuantizationParams<int8_t>(
      ranges[output_tensor_index][0], ranges[output_tensor_index][1]);
  tensors[output_tensor_index] = CreateQuantizedTensor<int8_t>(
      output, IntArrayFromInts(output_dim), quantization_params.scale,
      quantization_params.zero_point);

  int intermediate_dim[2] = {1, 0};
  for (int i = 0; i < 5; ++i) {
    tensors[intermediate_tensor_base + i] =
        CreateTensor<int16_t>(nullptr, IntArrayFromInts(intermediate_dim));
    intermediate_qparam[i].scale = FloatArrayFromFloats(intermediate_scale[i]);
    intermediate_qparam[i].zero_point = IntArrayFromInts(intermediate_zp[i]);
    intermediate_qparam[i].quantized_dimension = 0;
    tensors[intermediate_tensor_base + i].quantization.params =
        &intermediate_qparam[i];
  }

  TfLiteUnidirectionalSequenceLSTMParams params;
  params.activation = kTfLiteActTanh;
  params.cell_clip = 0.0f;
  params.proj_clip = 0.0f;
  params.time_major = time_major;
  params.asymmetric_quantize_inputs = asymmetric_quantize_inputs;

  const TfLiteRegistration registration =
      Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  micro::KernelRunner runner(
      registration, tensors, max_num_input_tensors + 1 + 5,
      IntArrayFromInts(inputs_array_data), IntArrayFromInts(outputs_array_data),
      reinterpret_cast<void*>(&params),
      IntArrayFromInts(intermediate_array_data));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
  for (int i = 0; i < sequence_length * n_batch * n_output; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output[i], output[i]);
  }
}

#endif  // !defined(XTENSA)

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)

TF_LITE_MICRO_TEST(UnidrectionalSequenceLstmIntegerNoPeepholeTest) {
  tflite::testing::TestUnidirectionalSequenceLstmInteger(
      tflite::testing::n_batch_integer_no_peephole,
      tflite::testing::n_input_integer_no_peephole,
      tflite::testing::n_cell_integer_no_peephole,
      tflite::testing::n_output_integer_no_peephole,
      tflite::testing::sequence_length_integer_no_peephole,
      /*time_major=*/true,
      /*use_cifg=*/false, /*use_peephole=*/false,
      /*use_projection_weights=*/true,
      /*use_projection_bias=*/false,
      /*use_layer_norm=*/true,
      /*use_8x8_8_implementation=*/false,
      tflite::testing::ranges_integer_no_peephole,
      tflite::testing::intermediate_scale_integer_no_peephole,
      tflite::testing::intermediate_zp_integer_no_peephole,
      tflite::testing::intermediate_qparam_integer_no_peephole,
      tflite::testing::lstm_input_integer_no_peephole,
      tflite::testing::lstm_input_integer_no_peephole_quantized,
      tflite::testing::input_to_input_weights_integer_no_peephole,
      tflite::testing::input_to_input_weights_integer_no_peephole_quantized,
      tflite::testing::input_to_forget_weights_integer_no_peephole,
      tflite::testing::input_to_forget_weights_integer_no_peephole_quantized,
      tflite::testing::input_to_cell_weights_integer_no_peephole,
      tflite::testing::input_to_cell_weights_integer_no_peephole_quantized,
      tflite::testing::input_to_output_weights_integer_no_peephole,
      tflite::testing::input_to_output_weights_integer_no_peephole_quantized,
      tflite::testing::recurrent_to_input_weights_integer_no_peephole,
      tflite::testing::recurrent_to_input_weights_integer_no_peephole_quantized,
      tflite::testing::recurrent_to_forget_weights_integer_no_peephole,
      tflite::testing::
          recurrent_to_forget_weights_integer_no_peephole_quantized,
      tflite::testing::recurrent_to_cell_weights_integer_no_peephole,
      tflite::testing::recurrent_to_cell_weights_integer_no_peephole_quantized,
      tflite::testing::recurrent_to_output_weights_integer_no_peephole,
      tflite::testing::
          recurrent_to_output_weights_integer_no_peephole_quantized,
      /*cell_to_input_weights=*/nullptr,
      /*cell_to_input_weights_quantized=*/nullptr,
      /*cell_to_forget_weights=*/nullptr,
      /*cell_to_forget_weights_quantized=*/nullptr,
      /*cell_to_output_weights=*/nullptr,
      /*cell_to_output_weights_quantized=*/nullptr,
      tflite::testing::input_gate_bias_integer_no_peephole,
      tflite::testing::input_gate_bias_integer_no_peephole_quantized,
      tflite::testing::forget_gate_bias_integer_no_peephole,
      tflite::testing::forget_gate_bias_integer_no_peephole_quantized,
      tflite::testing::cell_gate_bias_integer_no_peephole,
      tflite::testing::cell_gate_bias_integer_no_peephole_quantized,
      tflite::testing::output_gate_bias_integer_no_peephole,
      tflite::testing::output_gate_bias_integer_no_peephole_quantized,
      tflite::testing::projection_weights_integer_no_peephole,
      tflite::testing::projection_weights_integer_no_peephole_quantized,
      /*projection_bias=*/nullptr,
      /*projection_bias_quantized=*/nullptr,
      tflite::testing::output_state_no_peephole,
      tflite::testing::cell_state_no_peephole,
      tflite::testing::input_layer_norm_coefficients_integer_no_peephole,
      tflite::testing::
          input_layer_norm_coefficients_integer_no_peephole_quantized,
      tflite::testing::forget_layer_norm_coefficients_integer_no_peephole,
      tflite::testing::
          forget_layer_norm_coefficients_integer_no_peephole_quantized,
      tflite::testing::cell_layer_norm_coefficients_integer_no_peephole,
      tflite::testing::
          cell_layer_norm_coefficients_integer_no_peephole_quantized,
      tflite::testing::output_layer_norm_coefficients_integer_no_peephole,
      tflite::testing::
          output_layer_norm_coefficients_integer_no_peephole_quantized,
      tflite::testing::output_no_peephole,
      tflite::testing::expected_output_integer_no_peephole);
}

TF_LITE_MICRO_TEST(UnidrectionalSequenceLstmIntegerPeepholeTest) {
  tflite::testing::TestUnidirectionalSequenceLstmInteger(
      tflite::testing::n_batch_integer_peephole,
      tflite::testing::n_input_integer_peephole,
      tflite::testing::n_cell_integer_peephole,
      tflite::testing::n_output_integer_peephole,
      tflite::testing::sequence_length_integer_peephole,
      /*time_major=*/true,
      /*use_cifg=*/false, /*use_peephole=*/true,
      /*use_projection_weights=*/true,
      /*use_projection_bias=*/false,
      /*use_layer_norm=*/true,
      /*use_8x8_8_implementation=*/false,
      tflite::testing::ranges_integer_peephole,
      tflite::testing::intermediate_scale_integer_peephole,
      tflite::testing::intermediate_zp_integer_peephole,
      tflite::testing::intermediate_qparam_integer_peephole,
      tflite::testing::lstm_input_integer_peephole,
      tflite::testing::lstm_input_integer_peephole_quantized,
      tflite::testing::input_to_input_weights_integer_peephole,
      tflite::testing::input_to_input_weights_integer_peephole_quantized,
      tflite::testing::input_to_forget_weights_integer_peephole,
      tflite::testing::input_to_forget_weights_integer_peephole_quantized,
      tflite::testing::input_to_cell_weights_integer_peephole,
      tflite::testing::input_to_cell_weights_integer_peephole_quantized,
      tflite::testing::input_to_output_weights_integer_peephole,
      tflite::testing::input_to_output_weights_integer_peephole_quantized,
      tflite::testing::recurrent_to_input_weights_integer_peephole,
      tflite::testing::recurrent_to_input_weights_integer_peephole_quantized,
      tflite::testing::recurrent_to_forget_weights_integer_peephole,
      tflite::testing::recurrent_to_forget_weights_integer_peephole_quantized,
      tflite::testing::recurrent_to_cell_weights_integer_peephole,
      tflite::testing::recurrent_to_cell_weights_integer_peephole_quantized,
      tflite::testing::recurrent_to_output_weights_integer_peephole,
      tflite::testing::recurrent_to_output_weights_integer_peephole_quantized,
      tflite::testing::cell_to_input_weights_integer_peephole,
      tflite::testing::cell_to_input_weights_integer_peephole_quantized,
      tflite::testing::cell_to_forget_weights_integer_peephole,
      tflite::testing::cell_to_forget_weights_integer_peephole_quantized,
      tflite::testing::cell_to_output_weights_integer_peephole,
      tflite::testing::cell_to_output_weights_integer_peephole_quantized,
      tflite::testing::input_gate_bias_integer_peephole,
      tflite::testing::input_gate_bias_integer_peephole_quantized,
      tflite::testing::forget_gate_bias_integer_peephole,
      tflite::testing::forget_gate_bias_integer_peephole_quantized,
      tflite::testing::cell_gate_bias_integer_peephole,
      tflite::testing::cell_gate_bias_integer_peephole_quantized,
      tflite::testing::output_gate_bias_integer_peephole,
      tflite::testing::output_gate_bias_integer_peephole_quantized,
      tflite::testing::projection_weights_integer_peephole,
      tflite::testing::projection_weights_integer_peephole_quantized,
      /*projection_bias=*/nullptr,
      /*projection_bias_quantized=*/nullptr,
      tflite::testing::output_state_peephole,
      tflite::testing::cell_state_peephole,
      tflite::testing::input_layer_norm_coefficients_integer_peephole,
      tflite::testing::input_layer_norm_coefficients_integer_peephole_quantized,
      tflite::testing::forget_layer_norm_coefficients_integer_peephole,
      tflite::testing::
          forget_layer_norm_coefficients_integer_peephole_quantized,
      tflite::testing::cell_layer_norm_coefficients_integer_peephole,
      tflite::testing::cell_layer_norm_coefficients_integer_peephole_quantized,
      tflite::testing::output_layer_norm_coefficients_integer_peephole,
      tflite::testing::
          output_layer_norm_coefficients_integer_peephole_quantized,
      tflite::testing::output_peephole,
      tflite::testing::expected_output_integer_peephole);
}
#endif  // !defined(XTENSA)

TF_LITE_MICRO_TESTS_END
