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
#include "tensorflow/lite/micro/kernels/xtensa/lstm_eval.h"

#include <math.h>
#include <string.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {
namespace ops {
namespace micro {
namespace lstm_eval {
namespace {

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
#if !defined(HIFI5)
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
#endif  // !defined(HIFI5)
// Note: no aux_input.

// For each batch and cell: compute recurrent_weight * output_state.
#if !defined(HIFI5)
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
#endif  // !defined(HIFI5)
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
#if !defined(HIFI5)
      tensor_utils::PortableApplySigmoid(gate, n_batch, n_cell, gate);
#else
      xa_nn_vec_sigmoid_16_16(gate, gate, n_batch * n_cell);
#endif  // !defined(HIFI5)
      break;
    case kTfLiteActTanh:
#if !defined(HIFI5)
      tensor_utils::PortableApplyTanh(3, gate, n_batch, n_cell, gate);
#else
      xa_nn_vec_tanh_16_16(gate, gate, 3, n_batch * n_cell);
#endif  // !defined(HIFI5)
      break;
    default:
      // Only Sigmoid or Tanh is used.
      TFLITE_ASSERT_FALSE;
  }
}

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
#if !defined(HIFI5)
  // Use the forget_gate array as scratch, as input_gate array is not allocated
  // in CIFG case. (Be careful not to write to the scratch before reading the
  // forget gate data.)
  int16_t* scratch = forget_gate;

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
  tensor_utils::PortableCwiseAdd(cell_state, scratch, n_batch, n_cell,
                                 cell_state);

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

#endif  // !defined(HIFI5)
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
//  - output_state: output vector, size n_batch*n_output. Must be contiguous.
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
#if !defined(HIFI5)
  tensor_utils::PortableApplyTanh(15 + cell_state_scale, cell_state, n_batch,
                                  n_cell, scratch0);
#else
  xa_nn_vec_tanh_16_16(scratch0, cell_state, (15 + cell_state_scale),
                       n_batch * n_cell);
#endif  // !defined(HIFI5)

#if !defined(HIFI5)
  tensor_utils::PortableCwiseMul(output_gate, scratch0, hidden_scale_a,
                                 hidden_scale_b, n_batch, n_cell, hidden_zp,
                                 scratch1);
#else
  xa_nn_elm_mul_16x16_asym8s(scratch1, output_gate, scratch0, hidden_scale_a,
                             hidden_scale_b, hidden_zp, n_batch * n_cell);
#endif  // !defined(HIFI5)

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
//  - output_state: output vector, size n_batch*n_output. Must be contiguous.
//  - scratch: scratch area of size n_batch*n_cell
void CalculateLstmOutputInteger8x8_8(
    int n_batch, int n_cell, int n_output, const int16_t* cell_state,
    const int16_t* output_gate, const int8_t* projection_weights,
    int32_t proj_scale_a, int32_t proj_scale_b, const int32_t* projection_bias,
    int32_t output_state_zp, int32_t quantized_proj_clip, int8_t* output_state,
    int16_t* scratch) {
  // Note: unlike float/hybrid, the activation is always Tanh.
  tensor_utils::PortableApplyTanhFloat(cell_state, n_batch, n_cell, -15,
                                       scratch);
  tensor_utils::PortableCwiseMul(output_gate, scratch, n_batch, n_cell,
                                 15 + 15 - 15, scratch);
  // Note: no bias like in float/hybrid
  tensor_utils::PortableMatrixBatchVectorMultiply(
      scratch, projection_weights, proj_scale_a, proj_scale_b, projection_bias,
      n_batch, n_cell, n_output, output_state_zp, output_state);
  if (quantized_proj_clip > 0) {
    tensor_utils::PortableCwiseClipping(output_state, n_batch * n_output,
                                        (int8_t)quantized_proj_clip);
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
  // is always contiguous.
  std::copy_n(output_state_ptr, n_batch * n_output, output_ptr);
}

}  // namespace

// LINT.ThenChange(//tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc)
TfLiteStatus EvalInteger8x8_16(
    TfLiteContext* context, TfLiteNode* node, const TfLiteEvalTensor* input,
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
    const lstm_eval::IntegerLstmParameter* integer_lstm_param,
    TfLiteEvalTensor* output_state, TfLiteEvalTensor* cell_state,
    TfLiteEvalTensor* output, TfLiteEvalTensor* scratch0,
    TfLiteEvalTensor* scratch1, TfLiteEvalTensor* scratch2,
    TfLiteEvalTensor* scratch3, TfLiteEvalTensor* scratch4,
    TfLiteEvalTensor* scratch5) {
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

  if (time_major) {
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++) {
      const int t_rel = t;
      int8_t* output_ptr =
          tflite::micro::GetTensorData<int8_t>(output) + t_rel * output_step;
      const int8_t* input_ptr =
          tflite::micro::GetTensorData<int8_t>(input) + t_rel * input_step;
      LstmStepInteger8x8_16(
          input_ptr,
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
          output_state_zp, tflite::micro::GetTensorData<int16_t>(cell_state),
          output_ptr, (int16_t*)(scratch0), (int16_t*)(scratch1),
          (int16_t*)(scratch2), (int16_t*)(scratch3), (int8_t*)(scratch4),
          (int32_t*)(scratch5));
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
        const int8_t* input_ptr = tflite::micro::GetTensorData<int8_t>(input) +
                                  time_offset * input_step;
        int8_t* output_ptr = tflite::micro::GetTensorData<int8_t>(output) +
                             time_offset * output_step;

        // Offset the {output,cell}_state pointers to the right batch.
        int8_t* output_state_ptr =
            tflite::micro::GetTensorData<int8_t>(output_state) +
            b * output_batch_leading_dim;
        int16_t* cell_state_ptr =
            tflite::micro::GetTensorData<int16_t>(cell_state) + b * n_cell;

        LstmStepInteger8x8_16(
            input_ptr,
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
            tflite::micro::GetTensorData<int16_t>(
                input_layer_norm_coefficients),
            integer_lstm_param->layer_norm_input_scale_a,
            integer_lstm_param->layer_norm_input_scale_b,
            tflite::micro::GetTensorData<int16_t>(
                forget_layer_norm_coefficients),
            integer_lstm_param->layer_norm_forget_scale_a,
            integer_lstm_param->layer_norm_forget_scale_b,
            tflite::micro::GetTensorData<int16_t>(cell_layer_norm_coefficients),
            integer_lstm_param->layer_norm_cell_scale_a,
            integer_lstm_param->layer_norm_cell_scale_b,
            tflite::micro::GetTensorData<int16_t>(
                output_layer_norm_coefficients),
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
            cell_state_ptr, output_ptr, (int16_t*)(scratch0),
            (int16_t*)(scratch1), (int16_t*)(scratch2), (int16_t*)(scratch3),
            (int8_t*)(scratch4), (int32_t*)(scratch5));
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalInteger8x8_8(
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
    TfLiteEvalTensor* output_state, TfLiteEvalTensor* cell_state,
    TfLiteEvalTensor* output,
    const lstm_eval::IntegerLstmParameter* integer_lstm_param,
    TfLiteEvalTensor* scratch0, TfLiteEvalTensor* scratch1,
    TfLiteEvalTensor* scratch2, TfLiteEvalTensor* scratch3,
    TfLiteEvalTensor* scratch4, TfLiteEvalTensor* scratch5,
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
  // const int32_t input_zp = input->params.zero_point;
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
    int8_t* output_ptr =
        tflite::micro::GetTensorData<int8_t>(output) + t_rel * output_step;
    // Input can be int8 asymmetric or int16 symmetric.
    const int8_t* input_ptr =
        tflite::micro::GetTensorData<int8_t>(input) + t_rel * input_step;
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
        n_output, output_batch_leading_dim,
        tflite::micro::GetTensorData<int8_t>(output_state), output_state_zp,
        tflite::micro::GetTensorData<int16_t>(cell_state), output_ptr,
        tflite::micro::GetTensorData<int8_t>(scratch0),
        tflite::micro::GetTensorData<int8_t>(scratch1),
        tflite::micro::GetTensorData<int16_t>(scratch2),
        tflite::micro::GetTensorData<int16_t>(scratch3),
        tflite::micro::GetTensorData<int16_t>(scratch4),
        tflite::micro::GetTensorData<int16_t>(scratch5),
        tflite::micro::GetTensorData<int16_t>(scratch6),
        tflite::micro::GetTensorData<int16_t>(scratch7));
  }

  return kTfLiteOk;
}

}  // namespace lstm_eval
}  // namespace micro
}  // namespace ops
}  // namespace tflite
