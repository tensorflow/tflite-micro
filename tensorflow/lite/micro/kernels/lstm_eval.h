/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_H_

#include <cstdint>
#include <memory>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
// Since LSTM includes multiple intermediate stages, introducing the internal
// namespace to expose them for testing
namespace lstm_internal {
void CalculateLstmGateFloat(
    const float* input, const float* input_to_gate_weights,
    const float* aux_input, const float* aux_input_to_gate_weights,
    const float* output_state, const float* recurrent_to_gate_weights,
    const float* cell_state, const float* cell_to_gate_weights,
    const float* layer_norm_coefficients, const float* gate_bias,
    const int n_batch, const int n_input, const int n_aux_input,
    const int n_output, const int n_cell,
    const TfLiteFusedActivation activation, float* gate,
    const bool is_input_all_zeros, const bool is_aux_input_all_zeros);

void UpdateLstmCellFloat(int n_batch, int n_cell, float* cell_state,
                         const float* input_gate, float* forget_gate,
                         const float* cell_gate, bool use_cifg, float clip);

void CalculateLstmOutputFloat(int n_batch, int n_cell, int n_output,
                              const float* cell_state, const float* output_gate,
                              TfLiteFusedActivation activation,
                              const float* projection_weights,
                              const float* projection_bias,
                              const float proj_clip, float* output_state,
                              float* scratch);

void CalculateLstmGateInteger8x8_16(
    // Input and weights
    const int8_t* input, const int8_t* input_to_gate_weights,
    const int32_t* input_to_gate_bias, const int32_t input_to_gate_scale_a,
    const int32_t input_to_gate_scale_b,
    // Output state and weights
    const int8_t* output_state, const int8_t* recurrent_to_gate_weights,
    const int32_t* recurrent_to_gate_bias,
    const int32_t recurrent_to_gate_scale_a,
    const int32_t recurrent_to_gate_scale_b,
    // Cell state and weights
    const int16_t* cell_state, const int16_t* cell_to_gate_weights,
    const int32_t cell_to_gate_scale_a, const int32_t cell_to_gate_scale_b,
    // Layer normalization parameters (layer norm LSTM)
    const int16_t* layer_norm_coefficients, const int32_t* layer_norm_bias,
    const int32_t layer_norm_input_scale_a,
    const int32_t layer_norm_input_scale_b,
    const int32_t layer_norm_variance_guard,
    // Array sizes
    const int n_batch, const int n_input, const int n_output, const int n_cell,
    const TfLiteFusedActivation activation,
    // Output
    int16_t* gate,
    // Parameters for performance optimizations
    // Scratch arrays
    int32_t* scratch5);

void UpdateLstmCellInteger(int n_batch, int n_cell, int16_t* cell_state,
                           int32_t cell_state_scale, const int16_t* input_gate,
                           int16_t* forget_gate, const int16_t* cell_gate,
                           bool use_cifg, int16_t clip);

void CalculateLstmOutputInteger8x8_16(
    int n_batch, int n_cell, int n_output, int16_t* cell_state,
    int32_t cell_state_scale, const int16_t* output_gate,
    int32_t hidden_scale_a, int32_t hidden_scale_b, int32_t hidden_zp,
    const int8_t* projection_weights, int32_t proj_scale_a,
    int32_t proj_scale_b, const int32_t* projection_bias,
    int32_t output_state_zp, int8_t quantized_proj_clip, int8_t* output_state,
    int16_t* scratch0, int8_t* scratch1, int32_t* scratch2);

void LstmStepFloat(
    const float* input_ptr, const float* input_to_input_weights_ptr,
    const float* input_to_forget_weights_ptr,
    const float* input_to_cell_weights_ptr,
    const float* input_to_output_weights_ptr, const float* aux_input_ptr,
    const float* aux_input_to_input_weights_ptr,
    const float* aux_input_to_forget_weights_ptr,
    const float* aux_input_to_cell_weights_ptr,
    const float* aux_input_to_output_weights_ptr,
    const float* recurrent_to_input_weights_ptr,
    const float* recurrent_to_forget_weights_ptr,
    const float* recurrent_to_cell_weights_ptr,
    const float* recurrent_to_output_weights_ptr,
    const float* cell_to_input_weights_ptr,
    const float* cell_to_forget_weights_ptr,
    const float* cell_to_output_weights_ptr,
    const float* input_layer_norm_coefficients_ptr,
    const float* forget_layer_norm_coefficients_ptr,
    const float* cell_layer_norm_coefficients_ptr,
    const float* output_layer_norm_coefficients_ptr,
    const float* input_gate_bias_ptr, const float* forget_gate_bias_ptr,
    const float* cell_gate_bias_ptr, const float* output_gate_bias_ptr,
    const float* projection_weights_ptr, const float* projection_bias_ptr,
    const TfLiteLSTMParams* params, int n_batch, int n_cell, int n_input,
    int n_aux_input, int n_output, int output_batch_leading_dim,
    float* output_state_ptr, float* cell_state_ptr, float* scratch0,
    float* scratch1, float* scratch2, float* scratch3, float* output_ptr);

void LstmStepInteger8x8_16(
    const int8_t* input_ptr, const int8_t* input_to_input_weight_ptr,
    int32_t effective_input_to_input_scale_a,
    int32_t effective_input_to_input_scale_b,
    const int8_t* input_to_forget_weight_ptr,
    int32_t effective_input_to_forget_scale_a,
    int32_t effective_input_to_forget_scale_b,
    const int8_t* input_to_cell_weight_ptr,
    int32_t effective_input_to_cell_scale_a,
    int32_t effective_input_to_cell_scale_b,
    const int8_t* input_to_output_weight_ptr,
    int32_t effective_input_to_output_scale_a,
    int32_t effective_input_to_output_scale_b,
    const int8_t* recurrent_to_input_weight_ptr,
    int32_t effective_recurrent_to_input_scale_a,
    int32_t effective_recurrent_to_input_scale_b,
    const int8_t* recurrent_to_forget_weight_ptr,
    int32_t effective_recurrent_to_forget_scale_a,
    int32_t effective_recurrent_to_forget_scale_b,
    const int8_t* recurrent_to_cell_weight_ptr,
    int32_t effective_recurrent_to_cell_scale_a,
    int32_t effective_recurrent_to_cell_scale_b,
    const int8_t* recurrent_to_output_weight_ptr,
    int32_t effective_recurrent_to_output_scale_a,
    int32_t effective_recurrent_to_output_scale_b,
    const int16_t* cell_to_input_weight_ptr,
    int32_t effective_cell_to_input_scale_a,
    int32_t effective_cell_to_input_scale_b,
    const int16_t* cell_to_forget_weight_ptr,
    int32_t effective_cell_to_forget_scale_a,
    int32_t effective_cell_to_forget_scale_b,
    const int16_t* cell_to_output_weight_ptr,
    int32_t effective_cell_to_output_scale_a,
    int32_t effective_cell_to_output_scale_b,
    const int8_t* projection_weight_ptr, int32_t effective_proj_scale_a,
    int32_t effective_proj_scale_b, int32_t hidden_zp,
    int32_t effective_hidden_scale_a, int32_t effective_hidden_scale_b,
    const int16_t* layer_norm_input_weight_ptr,
    int32_t layer_norm_input_scale_a, int32_t layer_norm_input_scale_b,
    const int16_t* layer_norm_forget_weight_ptr,
    int32_t layer_norm_forget_scale_a, int32_t layer_norm_forget_scale_b,
    const int16_t* layer_norm_cell_weight_ptr, int32_t layer_norm_cell_scale_a,
    int32_t layer_norm_cell_scale_b,
    const int16_t* layer_norm_output_weight_ptr,
    int32_t layer_norm_output_scale_a, int32_t layer_norm_output_scale_b,
    const int32_t* input_gate_bias_ptr, const int32_t* forget_gate_bias_ptr,
    const int32_t* cell_gate_bias_ptr, const int32_t* output_gate_bias_ptr,
    int16_t quantized_cell_clip, int8_t quantized_proj_clip,
    int32_t cell_state_scale, int32_t input_variance_guard,
    int32_t forget_variance_guard, int32_t cell_variance_guard,
    int32_t output_variance_guard,
    const int32_t* input_to_forget_effective_bias,
    const int32_t* recurrent_to_forget_effective_bias,
    const int32_t* input_to_cell_effective_bias,
    const int32_t* recurrent_to_cell_effective_bias,
    const int32_t* input_to_output_effective_bias,
    const int32_t* recurrent_to_output_effective_bias,
    const int32_t* input_to_input_effective_bias,
    const int32_t* recurrent_to_input_effective_bias,
    const int32_t* projection_effective_bias, int n_batch, int n_cell,
    int n_input, int n_output, int8_t* output_state_ptr,
    int32_t output_state_zp, int16_t* cell_state_ptr, int8_t* output_ptr,
    int16_t* scratch0, int16_t* scratch1, int16_t* scratch2, int16_t* scratch3,
    int8_t* scratch4, int32_t* scratch5);
}  // namespace lstm_internal

// Pamameters for integer LSTM.
// Consider split this into two Integer Parameters if more fields are added.
struct IntegerLstmParameter {
  int32_t effective_input_to_input_scale_a = 0;
  int32_t effective_input_to_input_scale_b = 0;
  int32_t effective_recurrent_to_input_scale_a = 0;
  int32_t effective_recurrent_to_input_scale_b = 0;
  int32_t effective_cell_to_input_scale_a = 0;
  int32_t effective_cell_to_input_scale_b = 0;
  int32_t effective_input_to_forget_scale_a = 0;
  int32_t effective_input_to_forget_scale_b = 0;
  int32_t effective_recurrent_to_forget_scale_a = 0;
  int32_t effective_recurrent_to_forget_scale_b = 0;
  int32_t effective_cell_to_forget_scale_a = 0;
  int32_t effective_cell_to_forget_scale_b = 0;
  int32_t effective_input_to_cell_scale_a = 0;
  int32_t effective_input_to_cell_scale_b = 0;
  int32_t effective_recurrent_to_cell_scale_a = 0;
  int32_t effective_recurrent_to_cell_scale_b = 0;
  int32_t effective_input_to_output_scale_a = 0;
  int32_t effective_input_to_output_scale_b = 0;
  int32_t effective_recurrent_to_output_scale_a = 0;
  int32_t effective_recurrent_to_output_scale_b = 0;
  int32_t effective_cell_to_output_scale_a = 0;
  int32_t effective_cell_to_output_scale_b = 0;
  int32_t effective_proj_scale_a = 0;
  int32_t effective_proj_scale_b = 0;
  int32_t effective_hidden_scale_a = 0;
  int32_t effective_hidden_scale_b = 0;
  int32_t layer_norm_input_scale_a = 0;
  int32_t layer_norm_input_scale_b = 0;
  int32_t layer_norm_forget_scale_a = 0;
  int32_t layer_norm_forget_scale_b = 0;
  int32_t layer_norm_cell_scale_a = 0;
  int32_t layer_norm_cell_scale_b = 0;
  int32_t layer_norm_output_scale_a = 0;
  int32_t layer_norm_output_scale_b = 0;
  // Quantized clip value for cell and projection. Zero value means no
  // clipping.
  int16_t quantized_cell_clip = 0;
  int8_t quantized_proj_clip = 0;
  int32_t hidden_zp = 0;
  int32_t cell_scale = 0;

  int32_t input_variance_guard = 0;
  int32_t forget_variance_guard = 0;
  int32_t cell_variance_guard = 0;
  int32_t output_variance_guard = 0;

  // Pre-calculate bias + zero_point * weight.
  int32_t* input_to_forget_effective_bias = nullptr;
  int32_t* recurrent_to_forget_effective_bias = nullptr;
  int32_t* input_to_cell_effective_bias = nullptr;
  int32_t* recurrent_to_cell_effective_bias = nullptr;
  int32_t* input_to_output_effective_bias = nullptr;
  int32_t* recurrent_to_output_effective_bias = nullptr;
  int32_t* input_to_input_effective_bias = nullptr;
  int32_t* recurrent_to_input_effective_bias = nullptr;
  int32_t* projection_effective_bias = nullptr;

  // Scale and zero point for intermediate tensors.
  // Used only in the 8x8_8 case.
  int32_t intermediate_scale_a[8] = {};
  int32_t intermediate_scale_b[8] = {};
  int32_t intermediate_zp[12] = {};
};

TfLiteStatus EvalFloatLstm(
    const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* input_to_input_weights,
    const TfLiteEvalTensor* input_to_forget_weights,
    const TfLiteEvalTensor* input_to_cell_weights,
    const TfLiteEvalTensor* input_to_output_weights,
    const TfLiteEvalTensor* recurrent_to_input_weights,
    const TfLiteEvalTensor* recurrent_to_forget_weights,
    const TfLiteEvalTensor* recurrent_to_cell_weights,
    const TfLiteEvalTensor* recurrent_to_output_weights,
    const TfLiteEvalTensor* cell_to_input_weights,
    const TfLiteEvalTensor* cell_to_forget_weights,
    const TfLiteEvalTensor* cell_to_output_weights,
    const TfLiteEvalTensor* input_layer_norm_coefficients,
    const TfLiteEvalTensor* forget_layer_norm_coefficients,
    const TfLiteEvalTensor* cell_layer_norm_coefficients,
    const TfLiteEvalTensor* output_layer_norm_coefficients,
    const TfLiteEvalTensor* aux_input,
    const TfLiteEvalTensor* aux_input_to_input_weights,
    const TfLiteEvalTensor* aux_input_to_forget_weights,
    const TfLiteEvalTensor* aux_input_to_cell_weights,
    const TfLiteEvalTensor* aux_input_to_output_weights,
    const TfLiteEvalTensor* input_gate_bias,
    const TfLiteEvalTensor* forget_gate_bias,
    const TfLiteEvalTensor* cell_gate_bias,
    const TfLiteEvalTensor* output_gate_bias,
    const TfLiteEvalTensor* projection_weights,
    const TfLiteEvalTensor* projection_bias, const TfLiteLSTMParams* params,
    bool forward_sequence, bool time_major, int output_offset,
    float* scratch_buffer, TfLiteEvalTensor* output_state,
    TfLiteEvalTensor* cell_state, TfLiteEvalTensor* output);

TfLiteStatus EvalInteger8x8_16Lstm(
    const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* input_to_input_weights,
    const TfLiteEvalTensor* input_to_forget_weights,
    const TfLiteEvalTensor* input_to_cell_weights,
    const TfLiteEvalTensor* input_to_output_weights,
    const TfLiteEvalTensor* recurrent_to_input_weights,
    const TfLiteEvalTensor* recurrent_to_forget_weights,
    const TfLiteEvalTensor* recurrent_to_cell_weights,
    const TfLiteEvalTensor* recurrent_to_output_weights,
    const TfLiteEvalTensor* cell_to_input_weights,
    const TfLiteEvalTensor* cell_to_forget_weights,
    const TfLiteEvalTensor* cell_to_output_weights,
    const TfLiteEvalTensor* input_layer_norm_coefficients,
    const TfLiteEvalTensor* forget_layer_norm_coefficients,
    const TfLiteEvalTensor* cell_layer_norm_coefficients,
    const TfLiteEvalTensor* output_layer_norm_coefficients,
    const TfLiteEvalTensor* input_gate_bias,
    const TfLiteEvalTensor* forget_gate_bias,
    const TfLiteEvalTensor* cell_gate_bias,
    const TfLiteEvalTensor* output_gate_bias,
    const TfLiteEvalTensor* projection_weights,
    const TfLiteEvalTensor* projection_bias, const TfLiteLSTMParams* params,
    bool forward_sequence, bool time_major,
    const IntegerLstmParameter* integer_lstm_param, int32_t output_state_zp,
    TfLiteEvalTensor* output_state, TfLiteEvalTensor* cell_state,
    TfLiteEvalTensor* output, int16_t* scratch0, int16_t* scratch1,
    int16_t* scratch2, int16_t* scratch3, int8_t* scratch4, int32_t* scratch5);

}  // namespace tflite
#endif  // TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_H_
