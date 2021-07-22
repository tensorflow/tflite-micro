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
#include "tensorflow/lite/micro/kernels/xtensa/lstm/kernels/lstm_eval.h"

#include <math.h>
#include <string.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/op_macros.h"

//#define HIFI_NNLIB_OPT
//#define PROFILE
// #define PROFILE_SIGMOID
// #define PROFILE_TANH

#ifdef HIFI_NNLIB_OPT
#include <xtensa/config/core-isa.h>
#include <xtensa/tie/xt_core.h>
#include <xtensa/tie/xt_hifi2.h>
#include <xtensa/tie/xt_misc.h>

#include "xa_nnlib_api.h"
#endif

#ifdef PROFILE
#define PROF_ALLOCATE
#include <xtensa/config/core-isa.h>

#include "xt_profiler.h"
#endif

namespace tflite {
namespace ops {
namespace micro {
namespace lstm_eval {
namespace {
// LINT.IfChange
// Calculates a single LSTM gate.
//
// Implements the following formula: (* is matrix multiply)
//   gate = activate(W_input    * input + W_aux       * aux_input   +
//                   W_peephole * cell  + W_recurrent * prev_output + bias)
// with layer norm:
//   gate = activate(W_norm * normalize(...) + bias) // not adding bias inside
//
// Activation is sigmoid except for the "cell" gate (configurable, usually tanh)
//
// Parameters:
// Input vectors (to LSTM):    | Size:                | Optional?
//   input                     | n_input              |
//   aux_input                 | n_aux_input          | y (bidir LSTM)
// Input vectors (persistent states):
//   output_state              | n_output             |
//   cell_state                | n_cell               |
// 'Constant' inputs:
//   input_to_gate_weights     | n_cell * n_input     |
//   aux_input_to_gate_weights | n_cell * n_aux_input | y (bidir LSTM)
//   recurrent_to_gate_weights | n_cell * n_output    |
//   cell_to_gate_weights      | n_cell               | y (peephole)
//   gate_bias                 | n_cell               |
//   layer_norm_coefficients   | n_cell               | y (layer norm)
// Output vector:
//   gate                      | n_cell               |
// Scalar parameters:
//   n_batch                                    - batch size / number of vectors
//   n_input, n_aux_input, n_output, n_cell     - size of vectors.
//   activation                                 - activation to use.
//   is_input_all_zeros, is_aux_input_all_zeros - if input vectors are all zero.
//   use_layer_norm                             - if doing layer norm LSTM.
inline void CalculateLstmGateFloat(
    const float* input, const float* input_to_gate_weights,
    const float* aux_input, const float* aux_input_to_gate_weights,
    const float* output_state, const float* recurrent_to_gate_weights,
    const float* cell_state, const float* cell_to_gate_weights,
    const float* layer_norm_coefficients, const float* gate_bias,
    const int n_batch, const int n_input, const int n_aux_input,
    const int n_output, const int n_cell,
    const TfLiteFusedActivation activation, float* gate,
    const bool is_input_all_zeros, const bool is_aux_input_all_zeros) {
  const bool use_peephole = (cell_to_gate_weights != nullptr);
  const bool use_layer_norm = (layer_norm_coefficients != nullptr);

  // Initialize scratch buffers with bias for regular lstm or initialize with
  // zero for layer norm lstm.
  if (use_layer_norm) {
    std::fill_n(gate, n_cell * n_batch, 0.0f);
  } else {
    tensor_utils::VectorBatchVectorAssign(gate_bias, n_cell, n_batch, gate);
  }
  // For each batch and cell: compute input_weight * input.
  // Skip if input is all zeros.
  if (!is_input_all_zeros) {
    tensor_utils::PortableMatrixBatchVectorMultiplyAccumulate(
        input_to_gate_weights, n_cell, n_input, input, n_batch, gate);
  }
  // For each batch and cell: compute aux_input_weight * aux_input.
  // Skip if auxiliary input is not available or all zeros.
  if (!is_aux_input_all_zeros) {
    tensor_utils::PortableMatrixBatchVectorMultiplyAccumulate(aux_input_to_gate_weights,
                                                      n_cell, n_aux_input,
                                                      aux_input, n_batch, gate);
  }
  // For each batch and cell: compute recurrent_weight * output_state.
  tensor_utils::PortableMatrixBatchVectorMultiplyAccumulate(
      recurrent_to_gate_weights, n_cell, n_output, output_state, n_batch, gate);
  // For each batch and cell: compute cell_weight .* cell_state (peephole LSTM)
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_gate_weights, n_cell, cell_state, n_batch, gate);
  }
  // Do layer normalization (if layer norm LSTM)
  if (use_layer_norm) {
    tensor_utils::PortableMeanStddevNormalization(gate, gate, n_cell, n_batch);
    tensor_utils::VectorBatchVectorCwiseProduct(layer_norm_coefficients, n_cell,
                                                gate, n_batch, gate);
    tensor_utils::VectorBatchVectorAdd(gate_bias, n_cell, n_batch, gate);
  }
  // Apply activation
  //@TODO Need to enable this function for float implementation
 // tensor_utils::ApplyActivationToVector(gate, n_batch * n_cell, activation,
   //                                     gate);
}

// Updates the LSTM cell state, used by both float and hybrid LSTM versions.
//
// Implements the following formula:
//   cell_state_new = clip(forget_gate * cell_state + input_gate * cell_gate)
//
// With CIFG LSTM, input gate is replaced by (1-forget_gate).
//
// Parameters:
//  - n_batch, n_cell: sizes of vectors
//  - cell_state: input/output vector, size n_batch*n_cell
//  - input_gate: input vector, size n_batch*n_cell.
//  - forget_gate: input/scratch vector, size n_batch*n_cell, modified with CIFG
//  - cell_gate: input vector, size n_batch*n_cell.
//  - use_cifg: use 1-forget_gate instead of input_gate.
//  - clip: if > 0, clip the resulting cell state to [-clip, +clip].
void UpdateLstmCellFloat(int n_batch, int n_cell, float* cell_state,
                         const float* input_gate, float* forget_gate,
                         const float* cell_gate, bool use_cifg, float clip) {
  tensor_utils::VectorVectorCwiseProduct(forget_gate, cell_state,
                                         n_batch * n_cell, cell_state);

  if (use_cifg) {
    // With CIFG, input_gate = 1-forget_gate. Use the forget_gate array as
    // scratch, as input_gate array is not allocated in this case. (Be careful
    // not to write to the scratch before reading the forget gate data.)
    float* scratch = forget_gate;
    tensor_utils::PortableSub1Vector(forget_gate, n_batch * n_cell, scratch);
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_gate, scratch, n_batch * n_cell, cell_state);
  } else {
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_gate, input_gate, n_batch * n_cell, cell_state);
  }
  if (clip > 0.0f) {
    tensor_utils::PortableCwiseClipping(cell_state, n_batch * n_cell, clip);
  }
}

// Calculates the output state tensor of an LSTM step.
//
// Implements the following formula:
//   output_no_projection = output_gate .* activate(cell_state)
//     (elementwise vector product)
// If no projection is used:
//   output = output_state = output_no_projection
// With projection:
//   output = output_state = clip(W*output_no_projection + bias)
//
// Output might not have a different 'stride' than n_batch, so we need to copy.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - projection_weights, projection_weights_scale, projection_bias:
//      constant inputs, describing projection matrix and bias.
//  - proj_clip: if > 0, clip the output of the projection.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - scratch: scratch area, size n_batch*n_cell.
void CalculateLstmOutputFloat(int n_batch, int n_cell, int n_output,
                              const float* cell_state, const float* output_gate,
                              TfLiteFusedActivation activation,
                              const float* projection_weights,
                              const float* projection_bias,
                              const float proj_clip, float* output_state,
                              float* scratch) {
//@TODO Need to enable this function for float implementation
 // tensor_utils::ApplyActivationToVector(cell_state, n_batch * n_cell,
    ///                                    activation, scratch);
  tensor_utils::VectorVectorCwiseProduct(output_gate, scratch, n_batch * n_cell,
                                         scratch);

  const bool use_projection = (projection_weights != nullptr);
  const bool use_projection_bias = (projection_bias != nullptr);

  if (use_projection) {
    if (use_projection_bias) {
      tensor_utils::VectorBatchVectorAssign(projection_bias, n_output, n_batch,
                                            output_state);
    } else {
      std::fill_n(output_state, n_batch * n_output, 0.0f);
    }
    tensor_utils::PortableMatrixBatchVectorMultiplyAccumulate(
        projection_weights, n_output, n_cell, scratch, n_batch, output_state);
    if (proj_clip > 0.0f) {
      tensor_utils::PortableCwiseClipping(output_state, n_batch * n_output, proj_clip);
    }
  } else {
    std::copy_n(scratch, n_batch * n_output, output_state);
  }
}

// Calculates a single LSTM gate, int8x8_16 version.
// Implements the same functionality as CalculateLstmGateFloat.
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
    // CpuBackendContext* context,
    // Scratch arrays
    int32_t* scratch5) {
  const bool use_peephole = (cell_to_gate_weights != nullptr);
  const bool use_layer_norm = (layer_norm_coefficients != nullptr);

  // Initialize scratch buffers with zeros. Note that unlike float and hybrid
  // versions, bias is only used in layer normalization.
  std::fill_n(gate, n_batch * n_cell, 0);
#ifndef HIFI_NNLIB_OPT
  // For each batch and cell: compute input_weight * input.
  tensor_utils::PortableMatrixBatchVectorMultiplyAccumulate(
      input, input_to_gate_bias, input_to_gate_weights, input_to_gate_scale_a,
      input_to_gate_scale_b, n_batch, n_input, n_cell, 0, scratch5, gate, NULL);
#else
  {
    xa_nn_matXvec_acc_batch_sym8sx8_asym16s(
        gate, input_to_gate_weights, input, input_to_gate_bias, n_cell, n_input,
        n_input, input_to_gate_scale_a, input_to_gate_scale_b, 0, n_batch);
  }
#endif
// Note: no aux_input.

// For each batch and cell: compute recurrent_weight * output_state.
#ifndef HIFI_NNLIB_OPT
  tensor_utils::PortableMatrixBatchVectorMultiplyAccumulate(
      output_state, recurrent_to_gate_bias, recurrent_to_gate_weights,
      recurrent_to_gate_scale_a, recurrent_to_gate_scale_b, n_batch, n_output,
      n_cell, 0, scratch5, gate, NULL);
#else
  {
    xa_nn_matXvec_acc_batch_sym8sx8_asym16s(
        gate, recurrent_to_gate_weights, output_state, recurrent_to_gate_bias,
        n_cell, n_output, n_output, recurrent_to_gate_scale_a,
        recurrent_to_gate_scale_b, 0, n_batch);
  }
#endif
  // For each batch and cell: compute cell_weight * cell_state (peephole LSTM)
  if (use_peephole) {
    tensor_utils::PortableVectorBatchVectorCwiseProductAccumulate(
        cell_to_gate_weights, n_output, cell_state, n_batch,
        cell_to_gate_scale_a, cell_to_gate_scale_b, gate);
  }
  // Do layer normalization (if layer norm LSTM)
  if (use_layer_norm) {
    tensor_utils::PortableApplyLayerNorm(
        gate, layer_norm_coefficients, layer_norm_bias,
        layer_norm_input_scale_a, layer_norm_input_scale_b,
        layer_norm_variance_guard, n_batch, n_cell, gate);
  }
  // Apply activation
  switch (activation) {
    case kTfLiteActSigmoid:
#ifndef HIFI_NNLIB_OPT
      tensor_utils::PortableApplySigmoid(gate, n_batch, n_cell, gate);
#else
#ifdef VERIFY_NNLIB_SIGMOID
      int16_t inp[65536];
      int16_t tflm[65536], nnlib[65536];
      for (int it = -32768; it <= 32767; it++) inp[it + 32768] = it;
      tensor_utils::PortableApplySigmoid(inp, n_batch, 65536, tflm);
      xa_nn_vec_sigmoid_16_16(nnlib, inp, 65536);
      for (int it = 0; it < 65536; it++) {
        if (tflm[it] - nnlib[it] != 0)
          printf("Inp %d, tflm %d, nnlib %d \n", inp[it], tflm[it], nnlib[it]);
      }
      exit(0);
#else
#ifdef PROFILE_SIGMOID
      char profiler_sigmoid[MAX_PROFILER_PARAMS_LENGTH];
      sprintf(profiler_sigmoid, "Input Length %d,", n_batch * n_cell);
      XTPWR_PROFILER_OPEN(1, "sigmoid_16_16", profiler_sigmoid,
                          n_batch * n_cell, "cyc/point", 0);
      XTPWR_PROFILER_START(1);
#endif
      xa_nn_vec_sigmoid_16_16(gate, gate, n_batch * n_cell);
#ifdef PROFILE_SIGMOID
      XTPWR_PROFILER_STOP(1);
      XTPWR_PROFILER_UPDATE(1);
      XTPWR_PROFILER_PRINT(1);
      XTPWR_PROFILER_CLOSE(1, 1);
#endif
#endif
#endif
      break;
    case kTfLiteActTanh:
#ifndef HIFI_NNLIB_OPT
      tensor_utils::PortableApplyTanh(3, gate, n_batch, n_cell, gate);
#else
#ifdef PROFILE_TANH
      char profiler_tanh[MAX_PROFILER_PARAMS_LENGTH];
      sprintf(profiler_tanh, "Input Length %d,", n_batch * n_cell);
      XTPWR_PROFILER_OPEN(2, "tanh_16_16", profiler_tanh, n_batch * n_cell,
                          "cyc/point", 0);
      XTPWR_PROFILER_START(2);
#endif
      xa_nn_vec_tanh_16_16(gate, gate, 3, n_batch * n_cell);
#ifdef PROFILE_TANH
      XTPWR_PROFILER_STOP(2);
      XTPWR_PROFILER_UPDATE(2);
      XTPWR_PROFILER_PRINT(2);
      XTPWR_PROFILER_CLOSE(2, 1);
#endif
#endif
      break;
    default:
      // Only Sigmoid or Tanh is used.
      TFLITE_ASSERT_FALSE;
  }
}

#ifdef HIFI_NNLIB_OPT
void calc_cell_state_without_cifg(int16_t* cell_state,
                                  const int16_t* forget_gate,
                                  const int16_t* cell_gate,
                                  const int16_t* input_gate, int shift1,
                                  int shift2, int clip, int num_elms) {
  const ae_int16x8 *p16x8_cs_r, *p16x8_fg_r;
  const ae_int16x8 *p16x8_cg_r, *p16x8_ig_r;

  ae_int16x8* p16x8_cs_w;

  ae_valignx2 align_cs_r, align_fg_r;
  ae_valignx2 align_cg_r, align_ig_r;
  ae_valignx2 align_cs_w;

  ae_int16x4 d_cs_r_0, d_cs_r_1;
  ae_int16x4 d_fg_0, d_fg_1;
  ae_int16x4 d_cg_0, d_cg_1;
  ae_int16x4 d_ig_0, d_ig_1;
  ae_int16x4 d_cs_w_0, d_cs_w_1;
  ae_int32x2 d_mul_0, d_mul_1, d_mul_2, d_mul_3;
  ae_int32x2 d_mul_4, d_mul_5, d_mul_6, d_mul_7;

  ae_int16x4 d_min, d_max;

  int i = 0;
  p16x8_cs_r = (const ae_int16x8*)cell_state;
  p16x8_fg_r = (const ae_int16x8*)forget_gate;
  p16x8_cg_r = (const ae_int16x8*)cell_gate;
  p16x8_ig_r = (const ae_int16x8*)input_gate;

  p16x8_cs_w = (ae_int16x8*)cell_state;

  align_cs_r = AE_LA128_PP(p16x8_cs_r);
  align_fg_r = AE_LA128_PP(p16x8_fg_r);
  align_cg_r = AE_LA128_PP(p16x8_cg_r);
  align_ig_r = AE_LA128_PP(p16x8_ig_r);

  align_cs_w = AE_ZALIGN128();

  if (clip > 0) {
    d_min = AE_MOVDA16(-clip);
    d_max = AE_MOVDA16(clip);
  } else {
    d_min = AE_MOVDA16(-32768);
    d_max = AE_MOVDA16(32767);
  }

#pragma concurrent
  if (shift1 == 15) {
    for (i = 0; i < (num_elms >> 3); i++) {
      AE_LA16X4X2_IP(d_cs_r_0, d_cs_r_1, align_cs_r, p16x8_cs_r);
      AE_LA16X4X2_IP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r);
      AE_LA16X4X2_IP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r);
      AE_LA16X4X2_IP(d_ig_0, d_ig_1, align_ig_r, p16x8_ig_r);

      d_cs_w_0 = AE_MULFP16X4RS(d_cs_r_0, d_fg_0);
      d_cs_w_1 = AE_MULFP16X4RS(d_cs_r_1, d_fg_1);

      AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_ig_0);
      AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_ig_1);
      d_mul_4 = AE_SRAA32SYMS(d_mul_4, shift2);
      d_mul_5 = AE_SRAA32SYMS(d_mul_5, shift2);
      d_mul_6 = AE_SRAA32SYMS(d_mul_6, shift2);
      d_mul_7 = AE_SRAA32SYMS(d_mul_7, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
      d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      AE_MINMAX16(d_cs_w_1, d_min, d_max);

      AE_SA16X4X2_IP(d_cs_w_0, d_cs_w_1, align_cs_w, p16x8_cs_w);
    }
    AE_SA128POS_FP(align_cs_w, p16x8_cs_w);  // finalize the stream

    const ae_int16 *p16_cs_r, *p16_fg_r;
    const ae_int16 *p16_cg_r, *p16_ig_r;

    ae_int16* p16_cs_w;

    p16_cs_r = (const ae_int16*)p16x8_cs_r;
    p16_fg_r = (const ae_int16*)p16x8_fg_r;
    p16_cg_r = (const ae_int16*)p16x8_cg_r;
    p16_ig_r = (const ae_int16*)p16x8_ig_r;

    p16_cs_w = (ae_int16*)p16x8_cs_w;
// residue iterations
#pragma concurrent
#pragma loop_count max = 7
    for (i = 0; i < ((num_elms)&7); i++) {
      d_cs_r_0 = p16_cs_r[i];
      d_fg_0 = p16_fg_r[i];
      d_cg_0 = p16_cg_r[i];
      d_ig_0 = p16_ig_r[i];

      d_cs_w_0 = AE_MULFP16X4RS(d_cs_r_0, d_fg_0);

      AE_MUL16X4(d_mul_0, d_mul_1, d_cg_0, d_ig_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      p16_cs_w[i] = d_cs_w_0;
    }
  } else {
    for (i = 0; i < (num_elms >> 3); i++) {
      AE_LA16X4X2_IP(d_cs_r_0, d_cs_r_1, align_cs_r, p16x8_cs_r);
      AE_LA16X4X2_IP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r);
      AE_LA16X4X2_IP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r);
      AE_LA16X4X2_IP(d_ig_0, d_ig_1, align_ig_r, p16x8_ig_r);

      AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
      AE_MUL16X4(d_mul_2, d_mul_3, d_cs_r_1, d_fg_1);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift1);
      d_mul_1 = AE_SRAA32SYMS(d_mul_1, shift1);
      d_mul_2 = AE_SRAA32SYMS(d_mul_2, shift1);
      d_mul_3 = AE_SRAA32SYMS(d_mul_3, shift1);
      d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);
      d_cs_w_1 = AE_SAT16X4(d_mul_2, d_mul_3);

      AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_ig_0);
      AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_ig_1);
      d_mul_4 = AE_SRAA32SYMS(d_mul_4, shift2);
      d_mul_5 = AE_SRAA32SYMS(d_mul_5, shift2);
      d_mul_6 = AE_SRAA32SYMS(d_mul_6, shift2);
      d_mul_7 = AE_SRAA32SYMS(d_mul_7, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
      d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      AE_MINMAX16(d_cs_w_1, d_min, d_max);

      AE_SA16X4X2_IP(d_cs_w_0, d_cs_w_1, align_cs_w, p16x8_cs_w);
    }
    AE_SA128POS_FP(align_cs_w, p16x8_cs_w);  // finalize the stream

    const ae_int16 *p16_cs_r, *p16_fg_r;
    const ae_int16 *p16_cg_r, *p16_ig_r;

    ae_int16* p16_cs_w;

    p16_cs_r = (const ae_int16*)p16x8_cs_r;
    p16_fg_r = (const ae_int16*)p16x8_fg_r;
    p16_cg_r = (const ae_int16*)p16x8_cg_r;
    p16_ig_r = (const ae_int16*)p16x8_ig_r;

    p16_cs_w = (ae_int16*)p16x8_cs_w;
// residue iterations
#pragma concurrent
#pragma loop_count max = 7
    for (i = 0; i < ((num_elms)&7); i++) {
      d_cs_r_0 = p16_cs_r[i];
      d_fg_0 = p16_fg_r[i];
      d_cg_0 = p16_cg_r[i];
      d_ig_0 = p16_ig_r[i];

      AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift1);
      d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      AE_MUL16X4(d_mul_0, d_mul_1, d_cg_0, d_ig_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      p16_cs_w[i] = d_cs_w_0;
    }
  }
}

void calc_cell_state_with_cifg(int16_t* cell_state, const int16_t* forget_gate,
                               const int16_t* cell_gate, int shift1, int shift2,
                               int clip, int num_elms) {
  const ae_int16x8 *p16x8_cs_r, *p16x8_fg_r;
  const ae_int16x8* p16x8_cg_r;

  ae_int16x8* p16x8_cs_w;

  ae_valignx2 align_cs_r, align_fg_r;
  ae_valignx2 align_cg_r;
  ae_valignx2 align_cs_w;

  ae_int16x4 d_cs_r_0, d_cs_r_1;
  ae_int16x4 d_fg_0, d_fg_1;
  ae_int16x4 d_cg_0, d_cg_1;
  ae_int16x4 d_1mfg_0, d_1mfg_1;
  ae_int16x4 d_cs_w_0, d_cs_w_1;
  ae_int32x2 d_mul_0, d_mul_1, d_mul_2, d_mul_3;
  ae_int32x2 d_mul_4, d_mul_5, d_mul_6, d_mul_7;

  ae_int16x4 d_min, d_max, d_one;

  int i = 0;
  p16x8_cs_r = (const ae_int16x8*)cell_state;
  p16x8_fg_r = (const ae_int16x8*)forget_gate;
  p16x8_cg_r = (const ae_int16x8*)cell_gate;

  p16x8_cs_w = (ae_int16x8*)cell_state;

  align_cs_r = AE_LA128_PP(p16x8_cs_r);
  align_fg_r = AE_LA128_PP(p16x8_fg_r);
  align_cg_r = AE_LA128_PP(p16x8_cg_r);

  align_cs_w = AE_ZALIGN128();

  if (clip > 0) {
    d_min = AE_MOVDA16(-clip);
    d_max = AE_MOVDA16(clip);
  } else {
    d_min = AE_MOVDA16(-32768);
    d_max = AE_MOVDA16(32767);
  }
  d_one = AE_MOVDA16(32767);

#pragma concurrent
  if (shift1 == 15) {
    for (i = 0; i < (num_elms >> 3); i++) {
      AE_LA16X4X2_IP(d_cs_r_0, d_cs_r_1, align_cs_r, p16x8_cs_r);
      AE_LA16X4X2_IP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r);
      AE_LA16X4X2_IP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r);

      d_cs_w_0 = AE_MULFP16X4RS(d_cs_r_0, d_fg_0);
      d_cs_w_1 = AE_MULFP16X4RS(d_cs_r_1, d_fg_1);

      d_1mfg_0 = AE_SUB16S(d_one, d_fg_0);
      d_1mfg_1 = AE_SUB16S(d_one, d_fg_1);
      AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_1mfg_0);
      AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_1mfg_1);
      d_mul_4 = AE_SRAA32SYMS(d_mul_4, shift2);
      d_mul_5 = AE_SRAA32SYMS(d_mul_5, shift2);
      d_mul_6 = AE_SRAA32SYMS(d_mul_6, shift2);
      d_mul_7 = AE_SRAA32SYMS(d_mul_7, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
      d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      AE_MINMAX16(d_cs_w_1, d_min, d_max);

      AE_SA16X4X2_IP(d_cs_w_0, d_cs_w_1, align_cs_w, p16x8_cs_w);
    }
    AE_SA128POS_FP(align_cs_w, p16x8_cs_w);  // finalize the stream

    const ae_int16 *p16_cs_r, *p16_fg_r;
    const ae_int16 *p16_cg_r, *p16_ig_r;

    ae_int16* p16_cs_w;

    p16_cs_r = (const ae_int16*)p16x8_cs_r;
    p16_fg_r = (const ae_int16*)p16x8_fg_r;
    p16_cg_r = (const ae_int16*)p16x8_cg_r;

    p16_cs_w = (ae_int16*)p16x8_cs_w;
// residue iterations
#pragma concurrent
#pragma loop_count max = 7
    for (i = 0; i < ((num_elms)&7); i++) {
      d_cs_r_0 = p16_cs_r[i];
      d_fg_0 = p16_fg_r[i];
      d_cg_0 = p16_cg_r[i];

      d_cs_w_0 = AE_MULFP16X4RS(d_cs_r_0, d_fg_0);

      d_1mfg_0 = AE_SUB16S(d_one, d_fg_0);
      AE_MUL16X4(d_mul_0, d_mul_1, d_cg_0, d_1mfg_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      p16_cs_w[i] = d_cs_w_0;
    }
  } else {
    for (i = 0; i < (num_elms >> 3); i++) {
      AE_LA16X4X2_IP(d_cs_r_0, d_cs_r_1, align_cs_r, p16x8_cs_r);
      AE_LA16X4X2_IP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r);
      AE_LA16X4X2_IP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r);

      AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
      AE_MUL16X4(d_mul_2, d_mul_3, d_cs_r_1, d_fg_1);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift1);
      d_mul_1 = AE_SRAA32SYMS(d_mul_1, shift1);
      d_mul_2 = AE_SRAA32SYMS(d_mul_2, shift1);
      d_mul_3 = AE_SRAA32SYMS(d_mul_3, shift1);
      d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);
      d_cs_w_1 = AE_SAT16X4(d_mul_2, d_mul_3);

      d_1mfg_0 = AE_SUB16S(d_one, d_fg_0);
      d_1mfg_1 = AE_SUB16S(d_one, d_fg_1);
      AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_1mfg_0);
      AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_1mfg_1);
      d_mul_4 = AE_SRAA32SYMS(d_mul_4, shift2);
      d_mul_5 = AE_SRAA32SYMS(d_mul_5, shift2);
      d_mul_6 = AE_SRAA32SYMS(d_mul_6, shift2);
      d_mul_7 = AE_SRAA32SYMS(d_mul_7, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
      d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      AE_MINMAX16(d_cs_w_1, d_min, d_max);

      AE_SA16X4X2_IP(d_cs_w_0, d_cs_w_1, align_cs_w, p16x8_cs_w);
    }
    AE_SA128POS_FP(align_cs_w, p16x8_cs_w);  // finalize the stream

    const ae_int16 *p16_cs_r, *p16_fg_r;
    const ae_int16 *p16_cg_r, *p16_ig_r;

    ae_int16* p16_cs_w;

    p16_cs_r = (const ae_int16*)p16x8_cs_r;
    p16_fg_r = (const ae_int16*)p16x8_fg_r;
    p16_cg_r = (const ae_int16*)p16x8_cg_r;

    p16_cs_w = (ae_int16*)p16x8_cs_w;
// residue iterations
#pragma concurrent
#pragma loop_count max = 7
    for (i = 0; i < ((num_elms)&7); i++) {
      d_cs_r_0 = p16_cs_r[i];
      d_fg_0 = p16_fg_r[i];
      d_cg_0 = p16_cg_r[i];

      AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift1);
      d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      d_1mfg_0 = AE_SUB16S(d_one, d_fg_0);
      AE_MUL16X4(d_mul_0, d_mul_1, d_cg_0, d_1mfg_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      p16_cs_w[i] = d_cs_w_0;
    }
  }
}

void xa_nn_elm_mul_16x16_asym8s(int8_t* output, const int16_t* input_1,
                                const int16_t* input_2, int32_t multiplier,
                                int32_t shift, int32_t zero_point,
                                int num_elms) {
  ae_int16x8* tmp_input_1;
  ae_int16x8* tmp_input_2;

  ae_valignx2 align_src_input_1, align_src_input_2;
  ae_valign align_dst_output;

  ae_int16x4 data_a_0, data_a_1;
  ae_int16x4 data_b_0, data_b_1;
  ae_int32x2 data_ab_0, data_ab_1, data_ab_2, data_ab_3;
  ae_int32x2 d_multiplier, d_left_shift;
  ae_int16x4 d_zp;
  ae_int16x4 data_c_0, data_c_1;
  ae_int8x8 data_c;

  int i = 0;
  int left_shift, right_shift;
  tmp_input_1 = (ae_int16x8*)(input_1);
  tmp_input_2 = (ae_int16x8*)(input_2);

  align_src_input_1 = AE_LA128_PP((ae_int16x8*)tmp_input_1);
  align_src_input_2 = AE_LA128_PP((ae_int16x8*)tmp_input_2);
  align_dst_output = AE_ZALIGN64();  // zero alignment reg

  d_multiplier = AE_MOVDA32(multiplier);
  d_zp = AE_MOVDA16(zero_point);

  left_shift = shift < 0 ? 0 : shift;
  right_shift = shift > 0 ? 0 : -shift;

  d_left_shift = AE_MOVDA32(1 << left_shift);
#pragma concurrent
  for (i = 0; i < (num_elms >> 3); i++) {
    AE_LA16X4X2_IP(data_a_0, data_a_1, align_src_input_1, tmp_input_1);
    AE_LA16X4X2_IP(data_b_0, data_b_1, align_src_input_2, tmp_input_2);

    AE_MUL16X4(data_ab_0, data_ab_1, data_a_0, data_b_0);
    AE_MUL16X4(data_ab_2, data_ab_3, data_a_1, data_b_1);
    AE_MUL2P32X4(data_ab_0, data_ab_1, data_ab_0, data_ab_1, d_left_shift,
                 d_left_shift);
    AE_MUL2P32X4(data_ab_2, data_ab_3, data_ab_2, data_ab_3, d_left_shift,
                 d_left_shift);
    AE_MULF2P32X4RAS(data_ab_0, data_ab_1, data_ab_0, data_ab_1, d_multiplier,
                     d_multiplier);
    AE_MULF2P32X4RAS(data_ab_2, data_ab_3, data_ab_2, data_ab_3, d_multiplier,
                     d_multiplier);
    data_ab_0 = AE_SRAA32SYMS(data_ab_0, right_shift);
    data_ab_1 = AE_SRAA32SYMS(data_ab_1, right_shift);
    data_ab_2 = AE_SRAA32SYMS(data_ab_2, right_shift);
    data_ab_3 = AE_SRAA32SYMS(data_ab_3, right_shift);
    data_c_0 = AE_SAT16X4(data_ab_0, data_ab_1);
    data_c_1 = AE_SAT16X4(data_ab_2, data_ab_3);
    data_c_0 = AE_SUB16S(data_c_0, d_zp);
    data_c_1 = AE_SUB16S(data_c_1, d_zp);
    data_c = AE_SAT8X8X16(data_c_0, data_c_1);
    AE_SA8X8_IP(data_c, align_dst_output, (ae_int8x8*)output);
  }

  AE_SA64POS_FP(align_dst_output, output);  // finalize the stream

// residue iterations
#pragma concurrent
#pragma loop_count max = 7
  for (int j = 0; j < ((num_elms)&7); j++) {
    AE_L16_IP(data_a_0, (ae_int16*)tmp_input_1, 2);
    AE_L16_IP(data_b_0, (ae_int16*)tmp_input_2, 2);

    AE_MUL16X4(data_ab_0, data_ab_1, data_a_0, data_b_0);
    data_ab_0 = AE_MULP32X2(data_ab_0, d_left_shift);
    data_ab_0 = AE_MULFP32X2RAS(data_ab_0, d_multiplier);
    data_ab_0 = AE_SRAA32SYMS(data_ab_0, right_shift);
    data_c_0 = AE_SAT16X4(data_ab_0, data_ab_1);
    data_c_0 = AE_SUB16S(data_c_0, d_zp);
    data_c = AE_SAT8X8X16(data_c_0, data_c_0);
    AE_S8_0_IP(data_c, (ae_int8*)output, 1);
  }
}
#endif /* #ifdef HIFI_NNLIB_OPT */

// Updates the LSTM cell state, used by both integer LSTM versions.
// Also see UpdateLstmCellFloat.
//
// Parameters:
//  - n_batch, n_cell: sizes of vectors
//  - cell_state: input/output vector, size n_batch*n_cell
//  - cell_state_scale: scaling factor of cell state.
//  - input_gate: input vector, size n_batch*n_cell.
//  - forget_gate: input/scratch vector, size n_batch*n_cell, always modified.
//  - cell_gate: input vector, size n_batch*n_cell.
//  - use_cifg: use 1-forget_gate instead of input_gate.
//  - clip: if > 0, clip the resulting cell state to [-clip, +clip].
void UpdateLstmCellInteger(int n_batch, int n_cell, int16_t* cell_state,
                           int32_t cell_state_scale, const int16_t* input_gate,
                           int16_t* forget_gate, const int16_t* cell_gate,
                           bool use_cifg, int16_t clip) {
  // Use the forget_gate array as scratch, as input_gate array is not allocated
  // in CIFG case. (Be careful not to write to the scratch before reading the
  // forget gate data.)
  int16_t* scratch = forget_gate;

#ifndef HIFI_NNLIB_OPT
  tensor_utils::PortableCwiseMul(forget_gate, cell_state, n_batch, n_cell, 15,
                         cell_state);
  if (use_cifg) {
    tensor_utils::PortableSub1Vector(forget_gate, n_batch * n_cell, scratch);
    tensor_utils::PortableCwiseMul(scratch, cell_gate, n_batch, n_cell,
                           30 + cell_state_scale, scratch);
  } else {
    tensor_utils::PortableCwiseMul(input_gate, cell_gate, n_batch, n_cell,
                           30 + cell_state_scale, scratch);
  }
  tensor_utils::PortableCwiseAdd(cell_state, scratch, n_batch, n_cell, cell_state);

  if (clip > 0) {
    tensor_utils::PortableCwiseClipping(cell_state, n_batch * n_cell, clip);
  }
#else
  if (use_cifg) {
    calc_cell_state_with_cifg(cell_state, forget_gate, cell_gate, 15,
                              30 + cell_state_scale, clip, n_batch * n_cell);
  } else {
    calc_cell_state_without_cifg(cell_state, forget_gate, cell_gate, input_gate,
                                 15, 30 + cell_state_scale, clip,
                                 n_batch * n_cell);
  }

#endif
}

// Calculates the output state tensor of an LSTM step. See Float and hybrid
// versions as well.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - cell_state_scale: scaling of cell_state.
//  - hidden_scale_[a|b]: effective scale of cell_state.*output_gate
//  - hidden_zp: zero_point for cell_state.*output_gate
//  - projection_weights, proj_scale_[a|b], projection_bias:
//      constant inputs, describing projection matrix and bias.
//  - output_state_zp: zero point of output_state. (Input, calibrated value.)
//  - quantized_proj_clip: if > 0, clip the output of the projection.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - context: data for optimized MatrixBatchVectorMultiplyAccumulate.
//  - scratch0: scratch area of size n_batch*n_cell
//  - scratch1: scratch area of size n_batch*n_cell
//  - scratch2: scratch area used by MatrixBatchVectorMultiplyAccumulate
void CalculateLstmOutputInteger8x8_16(
    int n_batch, int n_cell, int n_output, const int16_t* cell_state,
    int32_t cell_state_scale, const int16_t* output_gate,
    int32_t hidden_scale_a, int32_t hidden_scale_b, int32_t hidden_zp,
    const int8_t* projection_weights, int32_t proj_scale_a,
    int32_t proj_scale_b, const int32_t* projection_bias,
    int32_t output_state_zp, int8_t quantized_proj_clip, int8_t* output_state,
    int16_t* scratch0, int8_t* scratch1, int32_t* scratch2) {
// Note: unlike float/hybrid, the activation is always Tanh.
#ifndef HIFI_NNLIB_OPT
  tensor_utils::PortableApplyTanh(15 + cell_state_scale, cell_state, n_batch, n_cell,
                          scratch0);
#else
#ifdef VERIFY_NNLIB_TANH
  int16_t inp[65536];
  int16_t tflm[65536], nnlib[65536];
  for (int j = 0; j < 7; j++) {
    for (int it = -32768; it <= 32767; it++) inp[it + 32768] = it;
    tensor_utils::PortableApplyTanh(j, inp, n_batch, 65536, tflm);
    xa_nn_vec_tanh_16_16(nnlib, inp, j, 65536);
    for (int it = 0; it < 65536; it++) {
      if (tflm[it] - nnlib[it] != 0)
        printf("Integer %d, Inp %d, tflm %d, nnlib %d \n", j, inp[it], tflm[it],
               nnlib[it]);
    }
  }
  exit(0);
#else
#ifdef PROFILE_TANH
  char profiler_tanh[MAX_PROFILER_PARAMS_LENGTH];
  sprintf(profiler_tanh, "Input Length %d,", n_batch * n_cell);
  XTPWR_PROFILER_OPEN(2, "tanh_16_16", profiler_tanh, n_batch * n_cell,
                      "cyc/point", 0);
  XTPWR_PROFILER_START(2);
#endif
  xa_nn_vec_tanh_16_16(scratch0, cell_state, (15 + cell_state_scale),
                       n_batch * n_cell);
#ifdef PROFILE_TANH
  XTPWR_PROFILER_STOP(2);
  XTPWR_PROFILER_UPDATE(2);
  XTPWR_PROFILER_PRINT(2);
  XTPWR_PROFILER_CLOSE(2, 1);
#endif
#endif
#endif

#ifndef HIFI_NNLIB_OPT
  tensor_utils::PortableCwiseMul(output_gate, scratch0, hidden_scale_a, hidden_scale_b,
                         n_batch, n_cell, hidden_zp, scratch1);
#else
  xa_nn_elm_mul_16x16_asym8s(scratch1, output_gate, scratch0, hidden_scale_a,
                             hidden_scale_b, hidden_zp, n_batch * n_cell);
#endif

  const bool use_projection = (projection_weights != nullptr);

  if (use_projection) {
    // Note: no bias like in float/hybrid
    std::fill_n(output_state, n_batch * n_output, 0);
    tensor_utils::PortableMatrixBatchVectorMultiplyAccumulate(
        scratch1, projection_bias, projection_weights, proj_scale_a,
        proj_scale_b, n_batch, n_cell, n_output, output_state_zp, scratch2,
        output_state, NULL);
    if (quantized_proj_clip > 0) {
      tensor_utils::PortableCwiseClipping(output_state, n_batch * n_output,
                                  quantized_proj_clip);
    }
  } else {
    std::copy_n(scratch1, n_batch * n_output, output_state);
  }
}

// Calculates a single LSTM gate, int8x8_8 version.
// Implements the same functionality as CalculateLstmGateFloat.
void CalculateLstmGateInteger8x8_8(
    // Inputs and weights
    const int8_t* input, int32_t input_zp, const int8_t* input_to_gate_weight,
    const int32_t input_to_gate_scale_a, const int32_t input_to_gate_scale_b,
    const int32_t input_times_weights_scale_a,
    const int32_t input_times_weights_scale_b,
    const int32_t input_times_weights_zp,
    // Output state and weights
    const int8_t* output_state, const int32_t output_state_zp,
    const int8_t* recurrent_to_gate_weight,
    const int32_t recurrent_to_gate_scale_a,
    const int32_t recurrent_to_gate_scale_b,
    const int32_t output_state_times_weights_scale_a,
    const int32_t output_state_times_weights_scale_b,
    const int32_t output_state_times_weights_zp,
    // Layer normalization parameters (layer norm LSTM)
    const int16_t* layer_norm_gate_weight,
    const int32_t layer_norm_gate_scale_a,
    const int32_t layer_norm_gate_scale_b, const int32_t* gate_bias,
    // Array sizes
    const int n_batch, const int n_input, const int n_output, const int n_cell,
    const TfLiteFusedActivation activation,
    // Output
    int16_t* gate,
    // Scratch arrays, both sized n_batch*n_cell
    int8_t* scratch0, int8_t* scratch1) {
  // Multiply input * input_weights => scratch0
  tensor_utils::PortableMatrixBatchVectorMultiply(
      input, input_zp, input_to_gate_weight, input_to_gate_scale_a,
      input_to_gate_scale_b, n_batch, n_input, n_cell, scratch0,
      input_times_weights_zp);
  // Multiply output_state * recurrent_weights => scratch1
  tensor_utils::PortableMatrixBatchVectorMultiply(
      output_state, output_state_zp, recurrent_to_gate_weight,
      recurrent_to_gate_scale_a, recurrent_to_gate_scale_b, n_batch, n_output,
      n_cell, scratch1, output_state_times_weights_zp);
  // Add scratch0 + scratch1 => gate
  tensor_utils::PortableTwoGateSaturatingAdd(
      scratch0, input_times_weights_zp, scratch1, output_state_times_weights_zp,
      input_times_weights_scale_a, input_times_weights_scale_b,
      output_state_times_weights_scale_a, output_state_times_weights_scale_b,
      n_batch, n_cell, gate);
  // Apply layer normalization.
  tensor_utils::PortableApplyLayerNormFloat(
      gate, layer_norm_gate_weight, layer_norm_gate_scale_a,
      layer_norm_gate_scale_b, gate_bias, n_batch, n_cell, gate);
  // Apply activation.
  switch (activation) {
    case kTfLiteActSigmoid:
      tensor_utils::PortableApplySigmoidFloat(gate, n_batch, n_cell, gate);
      break;
    case kTfLiteActTanh:
      tensor_utils::PortableApplyTanhFloat(gate, n_batch, n_cell, -12, gate);
      break;
    default:
      // Only Sigmoid or Tanh is used.
      TFLITE_ASSERT_FALSE;
  }
}

// Calculates the output state tensor of an LSTM step. See Float and hybrid
// versions as well.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - projection_weights, proj_scale_[a|b], projection_bias:
//      constant inputs, describing projection matrix and bias.
//  - output_state_zp: zero point of the output state.
//  - quantized_proj_clip: if > 0, clip the output of the projection.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - scratch: scratch area of size n_batch*n_cell
void CalculateLstmOutputInteger8x8_8(
    int n_batch, int n_cell, int n_output, const int16_t* cell_state,
    const int16_t* output_gate, const int8_t* projection_weights,
    int32_t proj_scale_a, int32_t proj_scale_b, const int32_t* projection_bias,
    int32_t output_state_zp, int32_t quantized_proj_clip, int8_t* output_state,
    int16_t* scratch) {
  // Note: unlike float/hybrid, the activation is always Tanh.
  tensor_utils::PortableApplyTanhFloat(cell_state, n_batch, n_cell, -15, scratch);
  tensor_utils::PortableCwiseMul(output_gate, scratch, n_batch, n_cell, 15 + 15 - 15,
                         scratch);
  // Note: no bias like in float/hybrid
  tensor_utils::PortableMatrixBatchVectorMultiply(
      scratch, projection_weights, proj_scale_a, proj_scale_b, projection_bias,
      n_batch, n_cell, n_output, output_state_zp, output_state);
  if (quantized_proj_clip > 0) {
      tensor_utils::PortableCwiseClipping(output_state, n_batch * n_output,
                            (int8_t)quantized_proj_clip);
  }
}

// Performs an LSTM batch inference step for input specified by input_ptr.
// The LSTM cell is specified by the pointers to its weights (*_weights_ptr) and
// biases (*_bias_ptr), and buffers (*_scratch), along with additional
// parameters:
//  - params: various LSTM params including activation, clipping, etc.,
//  - n_batch: size of batch,
//  - n_cell: number of cells (or units),
//  - n_input: the input size,
//  - n_aux_input: the auxiliary input size.
//  - n_output: the output size.
//  - output_batch_leading_dim: the leading dimension of the output buffer.
//
// Input of size 'n_batch * n_input':
//   input_ptr
// Input of size 'n_batch * n_aux_input':
//   aux_input_ptr                     - optional (can be nullptr)
//
// LSTM weights:
// Input weights of size 'n_cell * n_input':
//   input_to_input_weights            - optional
//   input_to_forget_weights
//   input_to_cell_weights
//   input_to_output_weights
// Auxiliary input weights of size 'n_cell * n_aux_input':
//   aux_input_to_input_weights        - optional
//   aux_input_to_forget_weights       - optional
//   aux_input_to_cell_weights         - optional
//   aux_input_to_output_weights       - optional
// Recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weights        - optional
//   recurrent_to_forget_weights
//   recurrent_to_cell_weights
//   recurrent_to_input_weights
// Peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights             - optional
//   cell_to_cell_weights              - optional
//   cell_to_output_weights            - optional
// Projection weights of size 'n_output * n_cell'
//   projection_weights_ptr            - optional
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr               - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Layer norm coefficients of size 'n_cell', representing diagonal matrices.
//   input_layer_norm_coefficients_ptr  - optional
//   forget_layer_norm_coefficients_ptr - optional
//   cell_layer_norm_coefficients_ptr   - optional
//   output_layer_norm_coefficients_ptr - optional
//
// The pointers to the cell and output state and the output are updated.
//
// The pointers input_ptr, aux_input_ptr, and output_ptr point to data aligned
// in batch_major order, and each step processes batch_size many inputs from
// input_ptr, and updates batch_size many cell and output states.
//
// The output_batch_dim is output.shape[-1], i.e. the outermost dimension of the
// output tensor, and in most cases will be equal to n_output. It is usually not
// when we want to store the LSTM output into a slice of the output tensor, e.g.
// for bidirectional LSTMs with merge_outputs. In this case, the batched
// operations cannot be used since they assume that the batched outputs are
// contiguous, and we manually loop over the batched outputs.
// LINT.IfChange
inline void LstmStepFloat(
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
    float* scratch1, float* scratch2, float* scratch3, float* output_ptr) {
  // ruy::profiler::ScopeLabel label("LstmStepFloat");
  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);

  // Make named scratch buffers.
  float* input_gate_scratch = scratch0;
  float* forget_gate_scratch = scratch1;
  float* cell_gate_scratch = scratch2;
  float* output_gate_scratch = scratch3;

  // Check if inputs are all zeros so we can skip some computations.
  const bool is_input_all_zeros =
      tensor_utils::PortableIsZeroVector(input_ptr, n_batch * n_input);
  const bool is_aux_input_all_zeros =
      (aux_input_ptr == nullptr ||
       tensor_utils::PortableIsZeroVector(aux_input_ptr, n_batch * n_aux_input));
  if (!use_cifg) {
    // Calculate the input gate. (If not CIFG.)
    CalculateLstmGateFloat(
        input_ptr, input_to_input_weights_ptr, aux_input_ptr,
        aux_input_to_input_weights_ptr, output_state_ptr,
        recurrent_to_input_weights_ptr, cell_state_ptr,
        cell_to_input_weights_ptr, input_layer_norm_coefficients_ptr,
        input_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
        /*activation=*/kTfLiteActSigmoid, input_gate_scratch,
        is_input_all_zeros, is_aux_input_all_zeros);
  }
  // Calculate the forget gate.
  CalculateLstmGateFloat(
      input_ptr, input_to_forget_weights_ptr, aux_input_ptr,
      aux_input_to_forget_weights_ptr, output_state_ptr,
      recurrent_to_forget_weights_ptr, cell_state_ptr,
      cell_to_forget_weights_ptr, forget_layer_norm_coefficients_ptr,
      forget_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
      /*activation=*/kTfLiteActSigmoid, forget_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros);
  // Calculate the cell update gate.
  CalculateLstmGateFloat(input_ptr, input_to_cell_weights_ptr, aux_input_ptr,
                         aux_input_to_cell_weights_ptr, output_state_ptr,
                         recurrent_to_cell_weights_ptr, /*cell_state=*/nullptr,
                         /*cell_to_gate_weights=*/nullptr,
                         cell_layer_norm_coefficients_ptr, cell_gate_bias_ptr,
                         n_batch, n_input, n_aux_input, n_output, n_cell,
                         params->activation, cell_gate_scratch,
                         is_input_all_zeros, is_aux_input_all_zeros);
  // Update the cell state.
  UpdateLstmCellFloat(n_batch, n_cell, cell_state_ptr, input_gate_scratch,
                      forget_gate_scratch, cell_gate_scratch, use_cifg,
                      params->cell_clip);
  // Calculate output gate.
  CalculateLstmGateFloat(
      input_ptr, input_to_output_weights_ptr, aux_input_ptr,
      aux_input_to_output_weights_ptr, output_state_ptr,
      recurrent_to_output_weights_ptr, cell_state_ptr,
      cell_to_output_weights_ptr, output_layer_norm_coefficients_ptr,
      output_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
      /*activation=*/kTfLiteActSigmoid, output_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros);
  // Update the output state.
  CalculateLstmOutputFloat(n_batch, n_cell, n_output, cell_state_ptr,
                           output_gate_scratch, params->activation,
                           projection_weights_ptr, projection_bias_ptr,
                           params->proj_clip, output_state_ptr, scratch2);
  // Copy output state to the output. Note that the output's rows may not be
  // contiguous (output_batch_leading_dim != n_output).
  for (int b = 0; b < n_batch; b++) {
    std::copy_n(output_state_ptr + b * n_output, n_output,
                output_ptr + b * output_batch_leading_dim);
  }
}

// Fully quantized lstm kernel for 16 bit gate matmul output.
//
// Input tensor of size n_batch * n_input:
//   input_ptr
//
// LSTM weights:
// Quantized input weights of size 'n_cell * n_input':
//   input_to_input_weight_ptr            - optional
//   input_to_forget_weight_ptr           - optional
//   input_to_cell_weight_ptr             - optional
//   input_to_output_weight_ptr           - optional
//
// Quantized recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weight_ptr        - optional
//   recurrent_to_forget_weights_ptr
//   recurrent_to_cell_weights_ptr
//   recurrent_to_input_weights_ptr
//
// Quantized peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights               - optional
//   cell_to_cell_weights                - optional
//   cell_to_output_weights              - optional
//
// Quantized projection weights of size 'n_output * n_cell'
//   projection_weight_ptr                     - optional
//
// Weight scales (scalars) for each of the weights above.
//   effective_input_to_input_scale_a    - optional
//   effective_input_to_input_scale_b    - optional
//   effective_input_to_forget_scale_a
//   effective_input_to_forget_scale_b
//   effective_input_to_cell_scale_a
//   effective_input_to_cell_scale_b
//   effective_input_to_output_scale_a
//   effective_input_to_output_scale_b
//   effective_recurrent_to_input_scale_a    - optional
//   effective_recurrent_to_input_scale_b    - optional
//   effective_recurrent_to_forget_scale_a
//   effective_recurrent_to_forget_scale_b
//   effective_recurrent_to_cell_scale_a
//   effective_recurrent_to_cell_scale_b
//   effective_recurrent_to_output_scale_a
//   effective_recurrent_to_output_scale_b
//   effective_proj_scale_a                  - optional
//   effective_proj_scale_b                  - optional
//
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr                 - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Layer norm coefficients of size 'n_cell', representing diagonal matrices.
//   layer_norm_input_weight_ptr    - optional
//   layer_norm_forget_weight_ptr   - optional
//   layer_norm_cell_weight_ptr     - optional
//   layer_norm_output_weight_ptr   - optional
//
// Layer norm scales of size 'n_cell'.
//   layer_norm_input_scale_a     - optional
//   layer_norm_input_scale_b     - optional
//   layer_norm_forget_scale_a    - optional
//   layer_norm_forget_scale_b    - optional
//   layer_norm_cell_scale_a      - optional
//   layer_norm_cell_scale_b      - optional
//   layer_norm_output_scale_a    - optional
//   layer_norm_output_scale_b    - optional
//
// Scalar values:
//   quantized_cell_clip: quantized clip value for cell.
//   quantized_proj_clip: quantized clip value for projection.
//   cell_state_scale: the power of two scale for cell state.
//
// Zero points:
//   output_state_zp: zero point of output state
//   hidden_zp: zero point for hidden state.
//
// Temporary pre-allocated storage for the calculation. Each is of size n_cell *
// n_batch.
//   scratch0
//   scratch1
//   scratch2
//   scratch3
//   scratch4
//   scratch5: this scratch buffer is created purely for optimizing the
//              MatrixBatchVectorMultiplyAccumulate.
//
// Outputs:
//   output_state_ptr - size 'n_batch * n_output'
//   cell_state_ptr   - size 'n_batch * n_cell'
//   output_ptr       - size 'n_batch * n_output'
// TODO(b/159947023): scratch0 is not used if (!cifg). Don't allocate then.
inline void LstmStepInteger8x8_16(
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
    int8_t* scratch4, int32_t* scratch5) {
  // ruy::profiler::ScopeLabel label("LstmStepInteger8x8_16");
  // Make named scratch buffers for the different gates.
  int16_t* input_gate_scratch = scratch0;
  int16_t* forget_gate_scratch = scratch1;
  int16_t* cell_gate_scratch = scratch2;
  int16_t* output_gate_scratch = scratch3;

#if 0  // LR_CHG
	memset( output_state_ptr , 0 , n_batch*n_output );
	memset( cell_state_ptr , 0 , n_batch*n_cell );
#endif

  // Since we have already checked that weights are all there or none, we
  // can check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weight_ptr == nullptr);

  // Check for nullptrs.
  TFLITE_DCHECK(input_to_forget_effective_bias);
  TFLITE_DCHECK(recurrent_to_forget_effective_bias);
  TFLITE_DCHECK(input_to_cell_effective_bias);
  TFLITE_DCHECK(recurrent_to_cell_effective_bias);
  TFLITE_DCHECK(input_to_output_effective_bias);
  TFLITE_DCHECK(recurrent_to_output_effective_bias);
  if (!use_cifg) {
    TFLITE_DCHECK(input_to_input_effective_bias);
    TFLITE_DCHECK(recurrent_to_input_effective_bias);
  }
  const bool use_projection = (projection_weight_ptr != nullptr);
  if (use_projection) {
    TFLITE_DCHECK(projection_effective_bias);
  }
  if (!use_cifg) {
    // Calculate the input gate. (If not CIFG.)
    CalculateLstmGateInteger8x8_16(
        input_ptr, input_to_input_weight_ptr, input_to_input_effective_bias,
        effective_input_to_input_scale_a, effective_input_to_input_scale_b,
        output_state_ptr, recurrent_to_input_weight_ptr,
        recurrent_to_input_effective_bias, effective_recurrent_to_input_scale_a,
        effective_recurrent_to_input_scale_b, cell_state_ptr,
        cell_to_input_weight_ptr, effective_cell_to_input_scale_a,
        effective_cell_to_input_scale_b, layer_norm_input_weight_ptr,
        input_gate_bias_ptr, layer_norm_input_scale_a, layer_norm_input_scale_b,
        input_variance_guard, n_batch, n_input, n_output, n_cell,
        kTfLiteActSigmoid, input_gate_scratch, scratch5);
  }
  // Calculate the forget gate.
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_forget_weight_ptr, input_to_forget_effective_bias,
      effective_input_to_forget_scale_a, effective_input_to_forget_scale_b,
      output_state_ptr, recurrent_to_forget_weight_ptr,
      recurrent_to_forget_effective_bias, effective_recurrent_to_forget_scale_a,
      effective_recurrent_to_forget_scale_b, cell_state_ptr,
      cell_to_forget_weight_ptr, effective_cell_to_forget_scale_a,
      effective_cell_to_forget_scale_b, layer_norm_forget_weight_ptr,
      forget_gate_bias_ptr, layer_norm_forget_scale_a,
      layer_norm_forget_scale_b, forget_variance_guard, n_batch, n_input,
      n_output, n_cell, kTfLiteActSigmoid, forget_gate_scratch, scratch5);
  // Calculate the cell update gate.
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_cell_weight_ptr, input_to_cell_effective_bias,
      effective_input_to_cell_scale_a, effective_input_to_cell_scale_b,
      output_state_ptr, recurrent_to_cell_weight_ptr,
      recurrent_to_cell_effective_bias, effective_recurrent_to_cell_scale_a,
      effective_recurrent_to_cell_scale_b, cell_state_ptr,
      /*cell_to_gate_weights=*/nullptr, /*cell_to_gate_scale_a=*/0,
      /*cell_to_gate_scale_b=*/0, layer_norm_cell_weight_ptr,
      cell_gate_bias_ptr, layer_norm_cell_scale_a, layer_norm_cell_scale_b,
      cell_variance_guard, n_batch, n_input, n_output, n_cell, kTfLiteActTanh,
      cell_gate_scratch, scratch5);
  // Update the cell state.
  UpdateLstmCellInteger(n_batch, n_cell, cell_state_ptr, cell_state_scale,
                        input_gate_scratch, forget_gate_scratch,
                        cell_gate_scratch, use_cifg, quantized_cell_clip);
  // Calculate the output gate.
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_output_weight_ptr, input_to_output_effective_bias,
      effective_input_to_output_scale_a, effective_input_to_output_scale_b,
      output_state_ptr, recurrent_to_output_weight_ptr,
      recurrent_to_output_effective_bias, effective_recurrent_to_output_scale_a,
      effective_recurrent_to_output_scale_b, cell_state_ptr,
      cell_to_output_weight_ptr, effective_cell_to_output_scale_a,
      effective_cell_to_output_scale_b, layer_norm_output_weight_ptr,
      output_gate_bias_ptr, layer_norm_output_scale_a,
      layer_norm_output_scale_b, output_variance_guard, n_batch, n_input,
      n_output, n_cell, kTfLiteActSigmoid, output_gate_scratch, scratch5);
  // Update the output state.
  CalculateLstmOutputInteger8x8_16(
      n_batch, n_cell, n_output, cell_state_ptr, cell_state_scale,
      output_gate_scratch, effective_hidden_scale_a, effective_hidden_scale_b,
      hidden_zp, projection_weight_ptr, effective_proj_scale_a,
      effective_proj_scale_b, projection_effective_bias, output_state_zp,
      quantized_proj_clip, output_state_ptr, scratch0, scratch4, scratch5);
  // Copy output state to the output. Note that unlike float or hybrid, output
  // is always contiguous.
  std::copy_n(output_state_ptr, n_batch * n_output, output_ptr);
}

// Fully quantized lstm kernel for 8 bit gate matmul output.
//
// Input tensor of size n_batch * n_input:
//   input_ptr
//
// LSTM weights:
// Quantized input weights of size 'n_cell * n_input':
//   input_to_input_weight_ptr            - optional
//   input_to_forget_weight_ptr           - optional
//   input_to_cell_weight_ptr             - optional
//   input_to_output_weight_ptr           - optional
//
// Quantized recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weight_ptr        - optional
//   recurrent_to_forget_weights_ptr
//   recurrent_to_cell_weights_ptr
//   recurrent_to_input_weights_ptr
//
// Quantized peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights               - optional
//   cell_to_cell_weights                - optional
//   cell_to_output_weights              - optional
//
// Quantized projection weights of size 'n_output * n_cell'
//   projection_weight_ptr                     - optional
//
// Weight scales (scalars) for each of the weights above.
//   effective_input_to_input_scale_a    - optional
//   effective_input_to_input_scale_b    - optional
//   effective_input_to_forget_scale_a
//   effective_input_to_forget_scale_b
//   effective_input_to_cell_scale_a
//   effective_input_to_cell_scale_b
//   effective_input_to_output_scale_a
//   effective_input_to_output_scale_b
//   effective_recurrent_to_input_scale_a    - optional
//   effective_recurrent_to_input_scale_b    - optional
//   effective_recurrent_to_forget_scale_a
//   effective_recurrent_to_forget_scale_b
//   effective_recurrent_to_cell_scale_a
//   effective_recurrent_to_cell_scale_b
//   effective_recurrent_to_output_scale_a
//   effective_recurrent_to_output_scale_b
//   effective_proj_scale_a                  - optional
//   effective_proj_scale_b                  - optional
//
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr                 - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Layer norm coefficients of size 'n_cell', representing diagonal matrices.
//   layer_norm_input_weight_ptr    - optional
//   layer_norm_forget_weight_ptr   - optional
//   layer_norm_cell_weight_ptr     - optional
//   layer_norm_output_weight_ptr   - optional
//
// Layer norm scales of size 'n_cell'.
//   layer_norm_input_scale_a     - optional
//   layer_norm_input_scale_b     - optional
//   layer_norm_forget_scale_a    - optional
//   layer_norm_forget_scale_b    - optional
//   layer_norm_cell_scale_a      - optional
//   layer_norm_cell_scale_b      - optional
//   layer_norm_output_scale_a    - optional
//   layer_norm_output_scale_b    - optional
//
// Scalar values:
//   quantized_cell_clip: quantized clip value for cell.
//   quantized_proj_clip: quantized clip value for projection.
//   cell_state_scale: the power of two scale for cell state.
//
// Zero points:
//   output_state_zp: zero point of output state.
//   hidden_zp: zero point for hidden state.
//
// Temporary pre-allocated storage for the calculation. Each is of size n_cell *
// n_batch.
//   scratch0
//   scratch1
//   scratch2
//   scratch3
//   scratch4
//   scratch5
//   scratch6
//   scratch7
//
// Outputs:
//   output_state_ptr - size 'n_batch * n_output'
//   cell_state_ptr   - size 'n_batch * n_cell'
//   output_ptr       - size 'n_batch * n_output'
// TODO(b/148688698): Move zero point calculation into Prepare().
// TODO(b/159947023): scratch5 is unused, remove.
inline void LstmStepInteger8x8_8(
    const int8_t* input_ptr, int32_t input_zp,
    const int8_t* input_to_input_weight_ptr,
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
    const int8_t* cell_to_input_weight_ptr,
    int32_t effective_cell_to_input_scale_a,
    int32_t effective_cell_to_input_scale_b,
    const int8_t* cell_to_forget_weight_ptr,
    int32_t effective_cell_to_forget_scale_a,
    int32_t effective_cell_to_forget_scale_b,
    const int8_t* cell_to_output_weight_ptr,
    int32_t effective_cell_to_output_scale_a,
    int32_t effective_cell_to_output_scale_b,
    const int8_t* projection_weight_ptr, int32_t effective_proj_scale_a,
    int32_t effective_proj_scale_b, const int16_t* layer_norm_input_weight_ptr,
    int32_t layer_norm_input_scale_a, int32_t layer_norm_input_scale_b,
    const int16_t* layer_norm_forget_weight_ptr,
    int32_t layer_norm_forget_scale_a, int32_t layer_norm_forget_scale_b,
    const int16_t* layer_norm_cell_weight_ptr, int32_t layer_norm_cell_scale_a,
    int32_t layer_norm_cell_scale_b,
    const int16_t* layer_norm_output_weight_ptr,
    int32_t layer_norm_output_scale_a, int32_t layer_norm_output_scale_b,
    const int32_t* input_gate_bias_ptr, const int32_t* forget_gate_bias_ptr,
    const int32_t* cell_gate_bias_ptr, const int32_t* output_gate_bias_ptr,
    const int32_t* projection_bias_ptr, const TfLiteLSTMParams* params,
    const int32_t* intermediate_scale_a, const int32_t* intermediate_scale_b,
    const int32_t* intermediate_zp, int16_t quantized_cell_clip,
    int8_t quantized_proj_clip, int n_batch, int n_cell, int n_input,
    int n_output, int output_batch_leading_dim, int8_t* output_state_ptr,
    int32_t output_state_zp, int16_t* cell_state_ptr, int8_t* output_ptr,
    int8_t* scratch0, int8_t* scratch1, int16_t* scratch2, int16_t* scratch3,
    int16_t* scratch4, int16_t* scratch5, int16_t* scratch6,
    int16_t* scratch7) {
  // TODO(b/159066113): scratch5 is unused, remove.

  // ruy::profiler::ScopeLabel label("LstmStepInteger8x8_8");
  // Make named scratch buffers for the different gates.
  int16_t* forget_gate_scratch = scratch2;
  int16_t* cell_gate_scratch = scratch3;
  int16_t* output_gate_scratch = scratch4;
  // no-CIFG is not supported here

  // Calculate the forget gate.
  CalculateLstmGateInteger8x8_8(
      input_ptr, input_zp, input_to_forget_weight_ptr,
      effective_input_to_forget_scale_a, effective_input_to_forget_scale_b,
      intermediate_scale_a[2], intermediate_scale_b[2], intermediate_zp[4],
      output_state_ptr, output_state_zp, recurrent_to_forget_weight_ptr,
      effective_recurrent_to_forget_scale_a,
      effective_recurrent_to_forget_scale_b, intermediate_scale_a[3],
      intermediate_scale_b[3], intermediate_zp[5], layer_norm_forget_weight_ptr,
      layer_norm_forget_scale_a, layer_norm_forget_scale_b,
      forget_gate_bias_ptr, n_batch, n_input, n_output, n_cell,
      kTfLiteActSigmoid, forget_gate_scratch, scratch0, scratch1);
  // Calculate the cell update gate.
  CalculateLstmGateInteger8x8_8(
      input_ptr, input_zp, input_to_cell_weight_ptr,
      effective_input_to_cell_scale_a, effective_input_to_cell_scale_b,
      intermediate_scale_a[4], intermediate_scale_b[4], intermediate_zp[7],
      output_state_ptr, output_state_zp, recurrent_to_cell_weight_ptr,
      effective_recurrent_to_cell_scale_a, effective_recurrent_to_cell_scale_b,
      intermediate_scale_a[5], intermediate_scale_b[5], intermediate_zp[8],
      layer_norm_cell_weight_ptr, layer_norm_cell_scale_a,
      layer_norm_cell_scale_b, cell_gate_bias_ptr, n_batch, n_input, n_output,
      n_cell, kTfLiteActTanh, cell_gate_scratch, scratch0, scratch1);
  // Update the cell state.
  UpdateLstmCellInteger(n_batch, n_cell, cell_state_ptr,
                        /*cell_state_scale=*/-15, /*input_gate=*/nullptr,
                        forget_gate_scratch, cell_gate_scratch,
                        /*use_cifg=*/true, quantized_cell_clip);
  // Calculate the output gate.
  CalculateLstmGateInteger8x8_8(
      input_ptr, input_zp, input_to_output_weight_ptr,
      effective_input_to_output_scale_a, effective_input_to_output_scale_b,
      intermediate_scale_a[6], intermediate_scale_b[6], intermediate_zp[10],
      output_state_ptr, output_state_zp, recurrent_to_output_weight_ptr,
      effective_recurrent_to_output_scale_a,
      effective_recurrent_to_output_scale_b, intermediate_scale_a[11],
      intermediate_scale_b[7], intermediate_zp[7], layer_norm_output_weight_ptr,
      layer_norm_output_scale_a, layer_norm_output_scale_b,
      output_gate_bias_ptr, n_batch, n_input, n_output, n_cell,
      kTfLiteActSigmoid, output_gate_scratch, scratch0, scratch1);
  // Update the output state.
  CalculateLstmOutputInteger8x8_8(
      n_batch, n_cell, n_output, cell_state_ptr, output_gate_scratch,
      projection_weight_ptr, effective_proj_scale_a, effective_proj_scale_b,
      projection_bias_ptr, output_state_zp, quantized_proj_clip,
      output_state_ptr, scratch2);
  // Copy output state to the output. Note that unlike float or hybrid, output
  // is always contigous.
  std::copy_n(output_state_ptr, n_batch * n_output, output_ptr);
}

}  // namespace

// LINT.IfChange
TfLiteStatus EvalFloat(
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* input_to_input_weights,
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
    const TfLiteEvalTensor* input_gate_bias, const TfLiteEvalTensor* forget_gate_bias,
    const TfLiteEvalTensor* cell_gate_bias, const TfLiteEvalTensor* output_gate_bias,
    const TfLiteEvalTensor* projection_weights, const TfLiteEvalTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, bool time_major,
    int output_offset, TfLiteEvalTensor* scratch_buffer, TfLiteEvalTensor* output_state,
    TfLiteEvalTensor* cell_state, TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->dims->size >= 2 && input->dims->size <= 3);
  int max_time, n_batch;
  if (input->dims->size == 3) {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  } else {
    max_time = 1;
    n_batch = input->dims->data[0];
  }
  const int n_input = input->dims->data[input->dims->size - 1];
  const int aux_input_size =
      (aux_input) ? aux_input->dims->data[aux_input->dims->size - 1] : 0;

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);

  // Index the scratch buffers pointers to the global scratch buffer.
  float* scratch_buffer_ptr = tflite::micro::GetTensorData<float>(scratch_buffer);
  float* input_gate_scratch = nullptr;
  float* cell_gate_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  if (use_cifg) {
    cell_gate_scratch = scratch_buffer_ptr;
    forget_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
  } else {
    input_gate_scratch = scratch_buffer_ptr;
    cell_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 3 * n_cell * n_batch;
  }

  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  if (time_major) {
    // Loop through the sequence.
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++) {
      // If this is the forward_sequence, step forward, otherwise step
      // backwards.
      const int t_rel = forward_sequence ? t : max_time - t - 1;
      const float* input_ptr = tflite::micro::GetTensorData<float>(input) + t_rel * input_step;
      const float* aux_input_ptr = nullptr;
      if (aux_input) {
        aux_input_ptr = tflite::micro::GetTensorData<float>(aux_input) + t_rel * input_step;
      }
      float* output_ptr =
          tflite::micro::GetTensorData<float>(output) + t_rel * output_step + output_offset;

      LstmStepFloat(
          input_ptr, tflite::micro::GetTensorData<float>(input_to_input_weights),
          tflite::micro::GetTensorData<float>(input_to_forget_weights),
          tflite::micro::GetTensorData<float>(input_to_cell_weights),
          tflite::micro::GetTensorData<float>(input_to_output_weights), aux_input_ptr,
          tflite::micro::GetTensorData<float>(aux_input_to_input_weights),
          tflite::micro::GetTensorData<float>(aux_input_to_forget_weights),
          tflite::micro::GetTensorData<float>(aux_input_to_cell_weights),
          tflite::micro::GetTensorData<float>(aux_input_to_output_weights),
          tflite::micro::GetTensorData<float>(recurrent_to_input_weights),
          tflite::micro::GetTensorData<float>(recurrent_to_forget_weights),
          tflite::micro::GetTensorData<float>(recurrent_to_cell_weights),
          tflite::micro::GetTensorData<float>(recurrent_to_output_weights),
          tflite::micro::GetTensorData<float>(cell_to_input_weights),
          tflite::micro::GetTensorData<float>(cell_to_forget_weights),
          tflite::micro::GetTensorData<float>(cell_to_output_weights),
          tflite::micro::GetTensorData<float>(input_layer_norm_coefficients),
          tflite::micro::GetTensorData<float>(forget_layer_norm_coefficients),
          tflite::micro::GetTensorData<float>(cell_layer_norm_coefficients),
          tflite::micro::GetTensorData<float>(output_layer_norm_coefficients),
          tflite::micro::GetTensorData<float>(input_gate_bias),
          tflite::micro::GetTensorData<float>(forget_gate_bias),
          tflite::micro::GetTensorData<float>(cell_gate_bias),
          tflite::micro::GetTensorData<float>(output_gate_bias),
          tflite::micro::GetTensorData<float>(projection_weights),
          tflite::micro::GetTensorData<float>(projection_bias), params, n_batch, n_cell,
          n_input, aux_input_size, n_output, output_batch_leading_dim,
          tflite::micro::GetTensorData<float>(output_state), tflite::micro::GetTensorData<float>(cell_state),
          input_gate_scratch, forget_gate_scratch, cell_gate_scratch,
          output_gate_scratch, output_ptr);
    }
  } else {
    for (int b = 0; b < n_batch; b++) {
      const int input_step = n_input;
      const int output_step = output_batch_leading_dim;
      for (int t = 0; t < max_time; t++) {
        // If this is the forward_sequence, step forward, otherwise step
        // backwards.
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        const int time_offset = b * max_time + t_rel;
        const float* input_ptr =
            tflite::micro::GetTensorData<float>(input) + time_offset * input_step;
        const float* aux_input_ptr = nullptr;
        if (aux_input) {
          aux_input_ptr =
              tflite::micro::GetTensorData<float>(aux_input) + time_offset * input_step;
        }
        float* output_ptr = tflite::micro::GetTensorData<float>(output) +
                            time_offset * output_step + output_offset;

        // Offset the {output,cell}_state pointers to the right batch.
        float* output_state_ptr =
            tflite::micro::GetTensorData<float>(output_state) + b * output_batch_leading_dim;
        float* cell_state_ptr = tflite::micro::GetTensorData<float>(cell_state) + b * n_cell;
        // Offset the scratch pointers to the right batch.
        float* input_gate_scratch_ptr =
            input_gate_scratch ? input_gate_scratch + b * n_cell : nullptr;
        float* forget_gate_scratch_ptr = forget_gate_scratch + b * n_cell;
        float* cell_gate_scratch_ptr = cell_gate_scratch + b * n_cell;
        float* output_gate_scratch_ptr = output_gate_scratch + b * n_cell;

        LstmStepFloat(
            input_ptr, tflite::micro::GetTensorData<float>(input_to_input_weights),
            tflite::micro::GetTensorData<float>(input_to_forget_weights),
            tflite::micro::GetTensorData<float>(input_to_cell_weights),
            tflite::micro::GetTensorData<float>(input_to_output_weights), aux_input_ptr,
            tflite::micro::GetTensorData<float>(aux_input_to_input_weights),
            tflite::micro::GetTensorData<float>(aux_input_to_forget_weights),
            tflite::micro::GetTensorData<float>(aux_input_to_cell_weights),
            tflite::micro::GetTensorData<float>(aux_input_to_output_weights),
            tflite::micro::GetTensorData<float>(recurrent_to_input_weights),
            tflite::micro::GetTensorData<float>(recurrent_to_forget_weights),
            tflite::micro::GetTensorData<float>(recurrent_to_cell_weights),
            tflite::micro::GetTensorData<float>(recurrent_to_output_weights),
            tflite::micro::GetTensorData<float>(cell_to_input_weights),
            tflite::micro::GetTensorData<float>(cell_to_forget_weights),
            tflite::micro::GetTensorData<float>(cell_to_output_weights),
            tflite::micro::GetTensorData<float>(input_layer_norm_coefficients),
            tflite::micro::GetTensorData<float>(forget_layer_norm_coefficients),
            tflite::micro::GetTensorData<float>(cell_layer_norm_coefficients),
            tflite::micro::GetTensorData<float>(output_layer_norm_coefficients),
            tflite::micro::GetTensorData<float>(input_gate_bias),
            tflite::micro::GetTensorData<float>(forget_gate_bias),
            tflite::micro::GetTensorData<float>(cell_gate_bias),
            tflite::micro::GetTensorData<float>(output_gate_bias),
            tflite::micro::GetTensorData<float>(projection_weights),
            tflite::micro::GetTensorData<float>(projection_bias), params, /*n_batch=*/1,
            n_cell, n_input, aux_input_size, n_output, output_batch_leading_dim,
            output_state_ptr, cell_state_ptr, input_gate_scratch_ptr,
            forget_gate_scratch_ptr, cell_gate_scratch_ptr,
            output_gate_scratch_ptr, output_ptr);
      }
    }
  }
  return kTfLiteOk;
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc)
TfLiteStatus EvalInteger8x8_16(
    TfLiteContext* context, TfLiteNode* node,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* input_to_input_weights,
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
    const TfLiteEvalTensor* input_gate_bias, const TfLiteEvalTensor* forget_gate_bias,
    const TfLiteEvalTensor* cell_gate_bias, const TfLiteEvalTensor* output_gate_bias,
    const TfLiteEvalTensor* projection_weights, const TfLiteEvalTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, bool time_major,
    const lstm_eval::IntegerLstmParameter* integer_lstm_param,
    TfLiteEvalTensor* output_state, TfLiteEvalTensor* cell_state, TfLiteEvalTensor* output,
    TfLiteEvalTensor* scratch0, TfLiteEvalTensor* scratch1, TfLiteEvalTensor* scratch2,
    TfLiteEvalTensor* scratch3, TfLiteEvalTensor* scratch4, TfLiteEvalTensor* scratch5) {
  TFLITE_DCHECK(input->dims->size >= 2 && input->dims->size <= 3);
  const int n_input = input->dims->data[input->dims->size - 1];
  int max_time, n_batch;
  if (input->dims->size == 2) {
    max_time = 1;
    n_batch = input->dims->data[0];
  } else {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  }

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Activation zero point
//  TODO@is data.output_zero_point equal to output_state->params.zero_point
  // int output_state_zp = output_state->params.zero_point;
  int output_state_zp = 0;

  // Get params for time/batch/sequence.
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];

#ifdef PROFILE
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH];
  sprintf(profiler_params, "Input %d, Output %d, Cell %d, Batch %d, MaxTime %d",
          n_input, n_output, n_cell, n_batch, max_time);
  XTPWR_PROFILER_OPEN(0, "LSTM_8X8_16", profiler_params, 0, "MACs/cycle", 0);
  XTPWR_PROFILER_START(0);
#endif

  if (time_major) {
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++) {
      const int t_rel = t;
      int8_t* output_ptr = tflite::micro::GetTensorData<int8_t>(output) + t_rel * output_step;
      const int8_t* input_ptr =
          tflite::micro::GetTensorData<int8_t>(input) + t_rel * input_step;
      LstmStepInteger8x8_16(
          input_ptr, tflite::micro::GetTensorData<int8_t>(input_to_input_weights),
          integer_lstm_param->effective_input_to_input_scale_a,
          integer_lstm_param->effective_input_to_input_scale_b,
          tflite::micro::GetTensorData<int8_t>(input_to_forget_weights),
          integer_lstm_param->effective_input_to_forget_scale_a,
          integer_lstm_param->effective_input_to_forget_scale_b,
          tflite::micro::GetTensorData<int8_t>(input_to_cell_weights),
          integer_lstm_param->effective_input_to_cell_scale_a,
          integer_lstm_param->effective_input_to_cell_scale_b,
          tflite::micro::GetTensorData<int8_t>(input_to_output_weights),
          integer_lstm_param->effective_input_to_output_scale_a,
          integer_lstm_param->effective_input_to_output_scale_b,
          tflite::micro::GetTensorData<int8_t>(recurrent_to_input_weights),
          integer_lstm_param->effective_recurrent_to_input_scale_a,
          integer_lstm_param->effective_recurrent_to_input_scale_b,
          tflite::micro::GetTensorData<int8_t>(recurrent_to_forget_weights),
          integer_lstm_param->effective_recurrent_to_forget_scale_a,
          integer_lstm_param->effective_recurrent_to_forget_scale_b,
          tflite::micro::GetTensorData<int8_t>(recurrent_to_cell_weights),
          integer_lstm_param->effective_recurrent_to_cell_scale_a,
          integer_lstm_param->effective_recurrent_to_cell_scale_b,
          tflite::micro::GetTensorData<int8_t>(recurrent_to_output_weights),
          integer_lstm_param->effective_recurrent_to_output_scale_a,
          integer_lstm_param->effective_recurrent_to_output_scale_b,
          tflite::micro::GetTensorData<int16_t>(cell_to_input_weights),
          integer_lstm_param->effective_cell_to_input_scale_a,
          integer_lstm_param->effective_cell_to_input_scale_b,
          tflite::micro::GetTensorData<int16_t>(cell_to_forget_weights),
          integer_lstm_param->effective_cell_to_forget_scale_a,
          integer_lstm_param->effective_cell_to_forget_scale_b,
          tflite::micro::GetTensorData<int16_t>(cell_to_output_weights),
          integer_lstm_param->effective_cell_to_output_scale_a,
          integer_lstm_param->effective_cell_to_output_scale_b,
          tflite::micro::GetTensorData<int8_t>(projection_weights),
          integer_lstm_param->effective_proj_scale_a,
          integer_lstm_param->effective_proj_scale_b,
          integer_lstm_param->hidden_zp,
          integer_lstm_param->effective_hidden_scale_a,
          integer_lstm_param->effective_hidden_scale_b,
          tflite::micro::GetTensorData<int16_t>(input_layer_norm_coefficients),
          integer_lstm_param->layer_norm_input_scale_a,
          integer_lstm_param->layer_norm_input_scale_b,
          tflite::micro::GetTensorData<int16_t>(forget_layer_norm_coefficients),
          integer_lstm_param->layer_norm_forget_scale_a,
          integer_lstm_param->layer_norm_forget_scale_b,
          tflite::micro::GetTensorData<int16_t>(cell_layer_norm_coefficients),
          integer_lstm_param->layer_norm_cell_scale_a,
          integer_lstm_param->layer_norm_cell_scale_b,
          tflite::micro::GetTensorData<int16_t>(output_layer_norm_coefficients),
          integer_lstm_param->layer_norm_output_scale_a,
          integer_lstm_param->layer_norm_output_scale_b,
          tflite::micro::GetTensorData<int32_t>(input_gate_bias),
          tflite::micro::GetTensorData<int32_t>(forget_gate_bias),
          tflite::micro::GetTensorData<int32_t>(cell_gate_bias),
          tflite::micro::GetTensorData<int32_t>(output_gate_bias),
          integer_lstm_param->quantized_cell_clip,
          integer_lstm_param->quantized_proj_clip,
          integer_lstm_param->cell_scale,
          integer_lstm_param->input_variance_guard,
          integer_lstm_param->forget_variance_guard,
          integer_lstm_param->cell_variance_guard,
          integer_lstm_param->output_variance_guard,
          integer_lstm_param->input_to_forget_effective_bias.get(),
          integer_lstm_param->recurrent_to_forget_effective_bias.get(),
          integer_lstm_param->input_to_cell_effective_bias.get(),
          integer_lstm_param->recurrent_to_cell_effective_bias.get(),
          integer_lstm_param->input_to_output_effective_bias.get(),
          integer_lstm_param->recurrent_to_output_effective_bias.get(),
          integer_lstm_param->input_to_input_effective_bias.get(),
          integer_lstm_param->recurrent_to_input_effective_bias.get(),
          integer_lstm_param->projection_effective_bias.get(), n_batch, n_cell,
          n_input, n_output, tflite::micro::GetTensorData<int8_t>(output_state),
          output_state_zp, tflite::micro::GetTensorData<int16_t>(cell_state), output_ptr,
          (int16_t *)(scratch0), (int16_t *)(scratch1),
          (int16_t *)(scratch2), (int16_t *)(scratch3),
          (int8_t *)(scratch4), (int32_t *)(scratch5));
    }
  } else {
    for (int b = 0; b < n_batch; b++) {
      const int input_step = n_input;
      const int output_step = output_batch_leading_dim;
      for (int t = 0; t < max_time; t++) {
        // If this is the forward_sequence, step forward, otherwise step
        // backwards.
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        const int time_offset = b * max_time + t_rel;
        const int8_t* input_ptr =
            tflite::micro::GetTensorData<int8_t>(input) + time_offset * input_step;
        int8_t* output_ptr =
            tflite::micro::GetTensorData<int8_t>(output) + time_offset * output_step;

        // Offset the {output,cell}_state pointers to the right batch.
        int8_t* output_state_ptr =
            tflite::micro::GetTensorData<int8_t>(output_state) + b * output_batch_leading_dim;
        int16_t* cell_state_ptr =
            tflite::micro::GetTensorData<int16_t>(cell_state) + b * n_cell;

        LstmStepInteger8x8_16(
            input_ptr, tflite::micro::GetTensorData<int8_t>(input_to_input_weights),
            integer_lstm_param->effective_input_to_input_scale_a,
            integer_lstm_param->effective_input_to_input_scale_b,
            tflite::micro::GetTensorData<int8_t>(input_to_forget_weights),
            integer_lstm_param->effective_input_to_forget_scale_a,
            integer_lstm_param->effective_input_to_forget_scale_b,
            tflite::micro::GetTensorData<int8_t>(input_to_cell_weights),
            integer_lstm_param->effective_input_to_cell_scale_a,
            integer_lstm_param->effective_input_to_cell_scale_b,
            tflite::micro::GetTensorData<int8_t>(input_to_output_weights),
            integer_lstm_param->effective_input_to_output_scale_a,
            integer_lstm_param->effective_input_to_output_scale_b,
            tflite::micro::GetTensorData<int8_t>(recurrent_to_input_weights),
            integer_lstm_param->effective_recurrent_to_input_scale_a,
            integer_lstm_param->effective_recurrent_to_input_scale_b,
            tflite::micro::GetTensorData<int8_t>(recurrent_to_forget_weights),
            integer_lstm_param->effective_recurrent_to_forget_scale_a,
            integer_lstm_param->effective_recurrent_to_forget_scale_b,
            tflite::micro::GetTensorData<int8_t>(recurrent_to_cell_weights),
            integer_lstm_param->effective_recurrent_to_cell_scale_a,
            integer_lstm_param->effective_recurrent_to_cell_scale_b,
            tflite::micro::GetTensorData<int8_t>(recurrent_to_output_weights),
            integer_lstm_param->effective_recurrent_to_output_scale_a,
            integer_lstm_param->effective_recurrent_to_output_scale_b,
            tflite::micro::GetTensorData<int16_t>(cell_to_input_weights),
            integer_lstm_param->effective_cell_to_input_scale_a,
            integer_lstm_param->effective_cell_to_input_scale_b,
            tflite::micro::GetTensorData<int16_t>(cell_to_forget_weights),
            integer_lstm_param->effective_cell_to_forget_scale_a,
            integer_lstm_param->effective_cell_to_forget_scale_b,
            tflite::micro::GetTensorData<int16_t>(cell_to_output_weights),
            integer_lstm_param->effective_cell_to_output_scale_a,
            integer_lstm_param->effective_cell_to_output_scale_b,
            tflite::micro::GetTensorData<int8_t>(projection_weights),
            integer_lstm_param->effective_proj_scale_a,
            integer_lstm_param->effective_proj_scale_b,
            integer_lstm_param->hidden_zp,
            integer_lstm_param->effective_hidden_scale_a,
            integer_lstm_param->effective_hidden_scale_b,
            tflite::micro::GetTensorData<int16_t>(input_layer_norm_coefficients),
            integer_lstm_param->layer_norm_input_scale_a,
            integer_lstm_param->layer_norm_input_scale_b,
            tflite::micro::GetTensorData<int16_t>(forget_layer_norm_coefficients),
            integer_lstm_param->layer_norm_forget_scale_a,
            integer_lstm_param->layer_norm_forget_scale_b,
            tflite::micro::GetTensorData<int16_t>(cell_layer_norm_coefficients),
            integer_lstm_param->layer_norm_cell_scale_a,
            integer_lstm_param->layer_norm_cell_scale_b,
            tflite::micro::GetTensorData<int16_t>(output_layer_norm_coefficients),
            integer_lstm_param->layer_norm_output_scale_a,
            integer_lstm_param->layer_norm_output_scale_b,
            tflite::micro::GetTensorData<int32_t>(input_gate_bias),
            tflite::micro::GetTensorData<int32_t>(forget_gate_bias),
            tflite::micro::GetTensorData<int32_t>(cell_gate_bias),
            tflite::micro::GetTensorData<int32_t>(output_gate_bias),
            integer_lstm_param->quantized_cell_clip,
            integer_lstm_param->quantized_proj_clip,
            integer_lstm_param->cell_scale,
            integer_lstm_param->input_variance_guard,
            integer_lstm_param->forget_variance_guard,
            integer_lstm_param->cell_variance_guard,
            integer_lstm_param->output_variance_guard,
            integer_lstm_param->input_to_forget_effective_bias.get(),
            integer_lstm_param->recurrent_to_forget_effective_bias.get(),
            integer_lstm_param->input_to_cell_effective_bias.get(),
            integer_lstm_param->recurrent_to_cell_effective_bias.get(),
            integer_lstm_param->input_to_output_effective_bias.get(),
            integer_lstm_param->recurrent_to_output_effective_bias.get(),
            integer_lstm_param->input_to_input_effective_bias.get(),
            integer_lstm_param->recurrent_to_input_effective_bias.get(),
            integer_lstm_param->projection_effective_bias.get(), /*n_batch=*/1,
            n_cell, n_input, n_output, output_state_ptr, output_state_zp,
            cell_state_ptr, output_ptr,
          (int16_t *)(scratch0), (int16_t *)(scratch1),
          (int16_t *)(scratch2), (int16_t *)(scratch3),
          (int8_t *)(scratch4), (int32_t *)(scratch5));
      }
    }
  }
#ifdef PROFILE
  XTPWR_PROFILER_STOP(0);
  XTPWR_PROFILER_UPDATE(0);
  XTPWR_PROFILER_PRINT(0);
  XTPWR_PROFILER_CLOSE(0, 1);
#endif

  return kTfLiteOk;
}

TfLiteStatus EvalInteger8x8_8(
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* input_to_input_weights,
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
    const TfLiteEvalTensor* input_gate_bias, const TfLiteEvalTensor* forget_gate_bias,
    const TfLiteEvalTensor* cell_gate_bias, const TfLiteEvalTensor* output_gate_bias,
    const TfLiteEvalTensor* projection_weights, const TfLiteEvalTensor* projection_bias,
    const TfLiteLSTMParams* params, TfLiteEvalTensor* output_state,
    TfLiteEvalTensor* cell_state, TfLiteEvalTensor* output,
    const lstm_eval::IntegerLstmParameter* integer_lstm_param,
    TfLiteEvalTensor* scratch0, TfLiteEvalTensor* scratch1, TfLiteEvalTensor* scratch2,
    TfLiteEvalTensor* scratch3, TfLiteEvalTensor* scratch4, TfLiteEvalTensor* scratch5,
    TfLiteEvalTensor* scratch6, TfLiteEvalTensor* scratch7) {
  TFLITE_DCHECK(input->dims->size >= 2 && input->dims->size <= 3);
  const int n_input = input->dims->data[input->dims->size - 1];
  int max_time, n_batch;
  if (input->dims->size == 2) {
    max_time = 1;
    n_batch = input->dims->data[0];
  } else {
    max_time = input->dims->data[0];
    n_batch = input->dims->data[1];
  }

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];
//@TODO input zero point and output zeropoint
  //const int32_t input_zp = input->params.zero_point;
 /// const int32_t output_state_zp = output_state->params.zero_point;
  const int32_t input_zp = 0;
  const int32_t output_state_zp = 0;

  // Get params for time/batch/sequence.
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  const int input_step = n_batch * n_input;
  const int output_step = n_batch * output_batch_leading_dim;

  for (int t = 0; t < max_time; t++) {
    const int t_rel = t;
    int8_t* output_ptr = tflite::micro::GetTensorData<int8_t>(output) + t_rel * output_step;
    // Input can be int8 asymmetric or int16 symmetric.
    const int8_t* input_ptr = tflite::micro::GetTensorData<int8_t>(input) + t_rel * input_step;
    lstm_eval::LstmStepInteger8x8_8(
        input_ptr, input_zp,

        tflite::micro::GetTensorData<int8_t>(input_to_input_weights),
        integer_lstm_param->effective_input_to_input_scale_a,
        integer_lstm_param->effective_input_to_input_scale_b,

        tflite::micro::GetTensorData<int8_t>(input_to_forget_weights),
        integer_lstm_param->effective_input_to_forget_scale_a,
        integer_lstm_param->effective_input_to_forget_scale_b,

        tflite::micro::GetTensorData<int8_t>(input_to_cell_weights),
        integer_lstm_param->effective_input_to_cell_scale_a,
        integer_lstm_param->effective_input_to_cell_scale_b,

        tflite::micro::GetTensorData<int8_t>(input_to_output_weights),
        integer_lstm_param->effective_input_to_output_scale_a,
        integer_lstm_param->effective_input_to_output_scale_b,

        tflite::micro::GetTensorData<int8_t>(recurrent_to_input_weights),
        integer_lstm_param->effective_recurrent_to_input_scale_a,
        integer_lstm_param->effective_recurrent_to_input_scale_b,

        tflite::micro::GetTensorData<int8_t>(recurrent_to_forget_weights),
        integer_lstm_param->effective_recurrent_to_forget_scale_a,
        integer_lstm_param->effective_recurrent_to_forget_scale_b,

        tflite::micro::GetTensorData<int8_t>(recurrent_to_cell_weights),
        integer_lstm_param->effective_recurrent_to_cell_scale_a,
        integer_lstm_param->effective_recurrent_to_cell_scale_b,

        tflite::micro::GetTensorData<int8_t>(recurrent_to_output_weights),
        integer_lstm_param->effective_recurrent_to_output_scale_a,
        integer_lstm_param->effective_recurrent_to_output_scale_b,

        tflite::micro::GetTensorData<int8_t>(cell_to_input_weights),
        integer_lstm_param->effective_cell_to_input_scale_a,
        integer_lstm_param->effective_cell_to_input_scale_b,

        tflite::micro::GetTensorData<int8_t>(cell_to_forget_weights),
        integer_lstm_param->effective_cell_to_forget_scale_a,
        integer_lstm_param->effective_cell_to_forget_scale_b,

        tflite::micro::GetTensorData<int8_t>(cell_to_output_weights),
        integer_lstm_param->effective_cell_to_output_scale_a,
        integer_lstm_param->effective_cell_to_output_scale_b,

        tflite::micro::GetTensorData<int8_t>(projection_weights),
        integer_lstm_param->effective_proj_scale_a,
        integer_lstm_param->effective_proj_scale_b,

        tflite::micro::GetTensorData<int16_t>(input_layer_norm_coefficients),
        integer_lstm_param->layer_norm_input_scale_a,
        integer_lstm_param->layer_norm_input_scale_b,

        tflite::micro::GetTensorData<int16_t>(forget_layer_norm_coefficients),
        integer_lstm_param->layer_norm_forget_scale_a,
        integer_lstm_param->layer_norm_forget_scale_b,

        tflite::micro::GetTensorData<int16_t>(cell_layer_norm_coefficients),
        integer_lstm_param->layer_norm_cell_scale_a,
        integer_lstm_param->layer_norm_cell_scale_b,

        tflite::micro::GetTensorData<int16_t>(output_layer_norm_coefficients),
        integer_lstm_param->layer_norm_output_scale_a,
        integer_lstm_param->layer_norm_output_scale_b,

        tflite::micro::GetTensorData<int32_t>(input_gate_bias),
        tflite::micro::GetTensorData<int32_t>(forget_gate_bias),
        tflite::micro::GetTensorData<int32_t>(cell_gate_bias),
        tflite::micro::GetTensorData<int32_t>(output_gate_bias),
        tflite::micro::GetTensorData<int32_t>(projection_bias),

        params, integer_lstm_param->intermediate_scale_a,
        integer_lstm_param->intermediate_scale_b,
        integer_lstm_param->intermediate_zp,
        integer_lstm_param->quantized_cell_clip,
        integer_lstm_param->quantized_proj_clip, n_batch, n_cell, n_input,
        n_output, output_batch_leading_dim, tflite::micro::GetTensorData<int8_t>(output_state),
        output_state_zp, tflite::micro::GetTensorData<int16_t>(cell_state), output_ptr,
        tflite::micro::GetTensorData<int8_t>(scratch0), tflite::micro::GetTensorData<int8_t>(scratch1),
        tflite::micro::GetTensorData<int16_t>(scratch2), tflite::micro::GetTensorData<int16_t>(scratch3),
        tflite::micro::GetTensorData<int16_t>(scratch4), tflite::micro::GetTensorData<int16_t>(scratch5),
        tflite::micro::GetTensorData<int16_t>(scratch6), tflite::micro::GetTensorData<int16_t>(scratch7));
  }

  return kTfLiteOk;
}

}  // namespace lstm_eval
}  // namespace micro
}  // namespace ops
}  // namespace tflite
