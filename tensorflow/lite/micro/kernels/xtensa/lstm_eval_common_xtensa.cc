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

#if defined(HIFI4) || defined(HIFI5)

#include <cmath>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/micro_tensor_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_lstm.h"

namespace tflite {

namespace {

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
//   gate_bias                 | n_cell               |
// Output vector:
//   gate                      | n_cell               |
// Scalar parameters:
//   n_batch                                    - batch size / number of vectors
//   n_input, n_aux_input, n_output, n_cell     - size of vectors.
//   activation                                 - activation to use.

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
    const int16_t* cell_state,
    // Array sizes
    const int n_batch, const int n_input, const int n_output, const int n_cell,
    const TfLiteFusedActivation activation,
    // Output
    int16_t* gate) {
  // Initialize scratch buffers with zeros. Note that unlike float and hybrid
  // versions, bias is only used in layer normalization.
  std::memset(gate, 0, n_batch * n_cell * sizeof(int16_t));
  xa_nn_matXvec_acc_batch_sym8sx8_asym16s(
      gate, input_to_gate_weights, input, input_to_gate_bias, n_cell, n_input,
      n_input, input_to_gate_scale_a, input_to_gate_scale_b, 0, n_batch);

  // Note: no aux_input.
  // For each batch and cell: compute recurrent_weight * output_state.
  xa_nn_matXvec_acc_batch_sym8sx8_asym16s(
      gate, recurrent_to_gate_weights, output_state, recurrent_to_gate_bias,
      n_cell, n_output, n_output, recurrent_to_gate_scale_a,
      recurrent_to_gate_scale_b, 0, n_batch);

  // Apply activation
  RuntimeShape gate_shape(1, n_batch * n_cell);
  switch (activation) {
    case kTfLiteActSigmoid: {
      lstm_internal::Sigmoid(gate_shape, gate);
    } break;
    case kTfLiteActTanh: {
      // Set the scale power to -12 to avoid shift
      lstm_internal::Tanh(/*cell_state_scale_power=*/-12, gate_shape, gate,
                          gate_shape, gate);
    } break;
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
//  - clip: if > 0, clip the resulting cell state to [-clip, +clip].
void UpdateLstmCellInteger(int n_batch, int n_cell, int16_t* cell_state,
                           int32_t cell_state_scale, const int16_t* input_gate,
                           int16_t* forget_gate, const int16_t* cell_gate,
                           int16_t clip) {
  lstm_xtensa::calc_cell_state_without_cifg(
      cell_state, forget_gate, cell_gate, input_gate, 15, 30 + cell_state_scale,
      clip, n_batch * n_cell);
}

// Calculates the output state tensor of an LSTM step.
//
// Implements the following formula:
//   output_no_projection = output_gate .* activate(cell_state)
//     (elementwise vector product)
// If no projection is used:
//   output = output_state = output_no_projection
//
// Output might not have a different 'stride' than n_batch, so we need to copy.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - scratch: scratch area, size n_batch*n_cell.

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
//  - output_state_zp: zero point of output_state. (Input, calibrated value.)
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - scratch0: scratch area of size n_batch*n_cell
//  - scratch1: scratch area of size n_batch*n_cell
//  - scratch2: scratch area used by MatrixBatchVectorMultiplyAccumulate
void CalculateLstmOutputInteger8x8_16(
    int n_batch, int n_cell, int n_output, int16_t* cell_state,
    int32_t cell_state_scale, const int16_t* output_gate,
    int32_t hidden_scale_a, int32_t hidden_scale_b, int32_t hidden_zp,
    int32_t output_state_zp, int8_t* output_state, int16_t* scratch0,
    int8_t* scratch1) {
  // Note: unlike float/hybrid, the activation is always Tanh.
  int32_t dims_data = n_batch * n_cell;
  RuntimeShape tanh_inp_shape(1, dims_data);
  lstm_internal::Tanh(cell_state_scale, tanh_inp_shape, cell_state,
                      tanh_inp_shape, scratch0);

  lstm_xtensa::xa_nn_elm_mul_16x16_asym8s(scratch1, output_gate, scratch0,
                                          hidden_scale_a, hidden_scale_b,
                                          hidden_zp, dims_data);

  std::memcpy(output_state, scratch1, dims_data * sizeof(int8_t));
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
//
// Scalar values:
//   quantized_cell_clip: quantized clip value for cell.
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
//
// Outputs:
//   output_state_ptr - size 'n_batch * n_output'
//   cell_state_ptr   - size 'n_batch * n_cell'
//   output_ptr       - size 'n_batch * n_output'
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
    int32_t effective_recurrent_to_output_scale_b, int32_t hidden_zp,
    int32_t effective_hidden_scale_a, int32_t effective_hidden_scale_b,
    int16_t quantized_cell_clip, int32_t cell_state_scale,
    const int32_t* input_to_forget_effective_bias,
    const int32_t* recurrent_to_forget_effective_bias,
    const int32_t* input_to_cell_effective_bias,
    const int32_t* recurrent_to_cell_effective_bias,
    const int32_t* input_to_output_effective_bias,
    const int32_t* recurrent_to_output_effective_bias,
    const int32_t* input_to_input_effective_bias,
    const int32_t* recurrent_to_input_effective_bias, int n_batch, int n_cell,
    int n_input, int n_output, int8_t* output_state_ptr,
    int32_t output_state_zp, int16_t* cell_state_ptr, int8_t* output_ptr,
    int16_t* scratch0, int16_t* scratch1, int16_t* scratch2, int16_t* scratch3,
    int8_t* scratch4) {
  // Make named scratch buffers for the different gates.
  int16_t* input_gate_scratch = scratch0;
  int16_t* forget_gate_scratch = scratch1;
  int16_t* cell_gate_scratch = scratch2;
  int16_t* output_gate_scratch = scratch3;

  // Check for nullptrs.
  TFLITE_DCHECK(input_to_forget_effective_bias);
  TFLITE_DCHECK(recurrent_to_forget_effective_bias);
  TFLITE_DCHECK(input_to_cell_effective_bias);
  TFLITE_DCHECK(recurrent_to_cell_effective_bias);
  TFLITE_DCHECK(input_to_output_effective_bias);
  TFLITE_DCHECK(recurrent_to_output_effective_bias);
  TFLITE_DCHECK(input_to_input_effective_bias);
  TFLITE_DCHECK(recurrent_to_input_effective_bias);

  // Calculate the input gate. (If not CIFG.)
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_input_weight_ptr, input_to_input_effective_bias,
      effective_input_to_input_scale_a, effective_input_to_input_scale_b,
      output_state_ptr, recurrent_to_input_weight_ptr,
      recurrent_to_input_effective_bias, effective_recurrent_to_input_scale_a,
      effective_recurrent_to_input_scale_b, cell_state_ptr, n_batch, n_input,
      n_output, n_cell, kTfLiteActSigmoid, input_gate_scratch);
  // Calculate the forget gate.
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_forget_weight_ptr, input_to_forget_effective_bias,
      effective_input_to_forget_scale_a, effective_input_to_forget_scale_b,
      output_state_ptr, recurrent_to_forget_weight_ptr,
      recurrent_to_forget_effective_bias, effective_recurrent_to_forget_scale_a,
      effective_recurrent_to_forget_scale_b, cell_state_ptr, n_batch, n_input,
      n_output, n_cell, kTfLiteActSigmoid, forget_gate_scratch);
  // Calculate the cell update gate.
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_cell_weight_ptr, input_to_cell_effective_bias,
      effective_input_to_cell_scale_a, effective_input_to_cell_scale_b,
      output_state_ptr, recurrent_to_cell_weight_ptr,
      recurrent_to_cell_effective_bias, effective_recurrent_to_cell_scale_a,
      effective_recurrent_to_cell_scale_b, cell_state_ptr, n_batch, n_input,
      n_output, n_cell, kTfLiteActTanh, cell_gate_scratch);
  // Update the cell state.
  UpdateLstmCellInteger(n_batch, n_cell, cell_state_ptr, cell_state_scale,
                        input_gate_scratch, forget_gate_scratch,
                        cell_gate_scratch, quantized_cell_clip);
  // Calculate the output gate.
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_output_weight_ptr, input_to_output_effective_bias,
      effective_input_to_output_scale_a, effective_input_to_output_scale_b,
      output_state_ptr, recurrent_to_output_weight_ptr,
      recurrent_to_output_effective_bias, effective_recurrent_to_output_scale_a,
      effective_recurrent_to_output_scale_b, cell_state_ptr, n_batch, n_input,
      n_output, n_cell, kTfLiteActSigmoid, output_gate_scratch);
  // Update the output state.
  CalculateLstmOutputInteger8x8_16(
      n_batch, n_cell, n_output, cell_state_ptr, cell_state_scale,
      output_gate_scratch, effective_hidden_scale_a, effective_hidden_scale_b,
      hidden_zp, output_state_zp, output_state_ptr, scratch0, scratch4);
  // Copy output state to the output. Note that unlike float or hybrid, output
  // is always contiguous.
  std::memcpy(output_ptr, output_state_ptr,
              n_batch * n_output * sizeof(int8_t));
}

}  // namespace

namespace lstm_xtensa {

TfLiteStatus EvalInteger8x8_16Lstm(
    const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* input_to_input_weights,
    const TfLiteEvalTensor* input_to_forget_weights,
    const TfLiteEvalTensor* input_to_cell_weights,
    const TfLiteEvalTensor* input_to_output_weights,
    const TfLiteEvalTensor* recurrent_to_input_weights,
    const TfLiteEvalTensor* recurrent_to_forget_weights,
    const TfLiteEvalTensor* recurrent_to_cell_weights,
    const TfLiteEvalTensor* recurrent_to_output_weights, bool forward_sequence,
    bool time_major, const XtensaOpDataLstm& op_data_lstm,
    TfLiteEvalTensor* output_state, TfLiteEvalTensor* cell_state,
    TfLiteEvalTensor* output, int16_t* scratch0, int16_t* scratch1,
    int16_t* scratch2, int16_t* scratch3, int8_t* scratch4) {
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
          tflite::micro::GetOptionalTensorData<int8_t>(input_to_input_weights),
          op_data_lstm.input_gate_parameters.input_fc_params.output_multiplier,
          op_data_lstm.input_gate_parameters.input_fc_params.output_shift,
          tflite::micro::GetOptionalTensorData<int8_t>(input_to_forget_weights),
          op_data_lstm.forget_gate_parameters.input_fc_params.output_multiplier,
          op_data_lstm.forget_gate_parameters.input_fc_params.output_shift,
          tflite::micro::GetOptionalTensorData<int8_t>(input_to_cell_weights),
          op_data_lstm.cell_gate_parameters.input_fc_params.output_multiplier,
          op_data_lstm.cell_gate_parameters.input_fc_params.output_shift,
          tflite::micro::GetOptionalTensorData<int8_t>(input_to_output_weights),
          op_data_lstm.output_gate_parameters.input_fc_params.output_multiplier,
          op_data_lstm.output_gate_parameters.input_fc_params.output_shift,
          tflite::micro::GetOptionalTensorData<int8_t>(
              recurrent_to_input_weights),
          op_data_lstm.input_gate_parameters.recurrent_fc_params
              .output_multiplier,
          op_data_lstm.input_gate_parameters.recurrent_fc_params.output_shift,
          tflite::micro::GetOptionalTensorData<int8_t>(
              recurrent_to_forget_weights),
          op_data_lstm.forget_gate_parameters.recurrent_fc_params
              .output_multiplier,
          op_data_lstm.forget_gate_parameters.recurrent_fc_params.output_shift,
          tflite::micro::GetOptionalTensorData<int8_t>(
              recurrent_to_cell_weights),
          op_data_lstm.cell_gate_parameters.recurrent_fc_params
              .output_multiplier,
          op_data_lstm.cell_gate_parameters.recurrent_fc_params.output_shift,
          tflite::micro::GetOptionalTensorData<int8_t>(
              recurrent_to_output_weights),
          op_data_lstm.output_gate_parameters.recurrent_fc_params
              .output_multiplier,
          op_data_lstm.output_gate_parameters.recurrent_fc_params.output_shift,
          op_data_lstm.inter_gate_parameters.output_mul_params.output_offset,
          op_data_lstm.inter_gate_parameters.output_mul_params
              .output_multiplier,
          op_data_lstm.inter_gate_parameters.output_mul_params.output_shift,
          op_data_lstm.cell_state_info.quantized_cell_clip,
          op_data_lstm.cell_state_info.cell_state_scale_power,
          op_data_lstm.integer_lstm_param.input_to_forget_effective_bias,
          op_data_lstm.integer_lstm_param.recurrent_to_forget_effective_bias,
          op_data_lstm.integer_lstm_param.input_to_cell_effective_bias,
          op_data_lstm.integer_lstm_param.recurrent_to_cell_effective_bias,
          op_data_lstm.integer_lstm_param.input_to_output_effective_bias,
          op_data_lstm.integer_lstm_param.recurrent_to_output_effective_bias,
          op_data_lstm.integer_lstm_param.input_to_input_effective_bias,
          op_data_lstm.integer_lstm_param.recurrent_to_input_effective_bias,
          n_batch, n_cell, n_input, n_output,
          tflite::micro::GetTensorData<int8_t>(output_state),
          op_data_lstm.inter_gate_parameters.output_mul_params.output_offset,
          tflite::micro::GetTensorData<int16_t>(cell_state), output_ptr,
          scratch0, scratch1, scratch2, scratch3, scratch4);
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
            tflite::micro::GetOptionalTensorData<int8_t>(
                input_to_input_weights),
            op_data_lstm.input_gate_parameters.input_fc_params
                .output_multiplier,
            op_data_lstm.input_gate_parameters.input_fc_params.output_shift,
            tflite::micro::GetOptionalTensorData<int8_t>(
                input_to_forget_weights),
            op_data_lstm.forget_gate_parameters.input_fc_params
                .output_multiplier,
            op_data_lstm.forget_gate_parameters.input_fc_params.output_shift,
            tflite::micro::GetOptionalTensorData<int8_t>(input_to_cell_weights),
            op_data_lstm.cell_gate_parameters.input_fc_params.output_multiplier,
            op_data_lstm.cell_gate_parameters.input_fc_params.output_shift,
            tflite::micro::GetOptionalTensorData<int8_t>(
                input_to_output_weights),
            op_data_lstm.output_gate_parameters.input_fc_params
                .output_multiplier,
            op_data_lstm.output_gate_parameters.input_fc_params.output_shift,
            tflite::micro::GetOptionalTensorData<int8_t>(
                recurrent_to_input_weights),
            op_data_lstm.input_gate_parameters.recurrent_fc_params
                .output_multiplier,
            op_data_lstm.input_gate_parameters.recurrent_fc_params.output_shift,
            tflite::micro::GetOptionalTensorData<int8_t>(
                recurrent_to_forget_weights),
            op_data_lstm.forget_gate_parameters.recurrent_fc_params
                .output_multiplier,
            op_data_lstm.forget_gate_parameters.recurrent_fc_params
                .output_shift,
            tflite::micro::GetOptionalTensorData<int8_t>(
                recurrent_to_cell_weights),
            op_data_lstm.cell_gate_parameters.recurrent_fc_params
                .output_multiplier,
            op_data_lstm.cell_gate_parameters.recurrent_fc_params.output_shift,
            tflite::micro::GetOptionalTensorData<int8_t>(
                recurrent_to_output_weights),
            op_data_lstm.output_gate_parameters.recurrent_fc_params
                .output_multiplier,
            op_data_lstm.output_gate_parameters.recurrent_fc_params
                .output_shift,
            op_data_lstm.inter_gate_parameters.output_mul_params.output_offset,
            op_data_lstm.inter_gate_parameters.output_mul_params
                .output_multiplier,
            op_data_lstm.inter_gate_parameters.output_mul_params.output_shift,
            op_data_lstm.cell_state_info.quantized_cell_clip,
            op_data_lstm.cell_state_info.cell_state_scale_power,
            op_data_lstm.integer_lstm_param.input_to_forget_effective_bias,
            op_data_lstm.integer_lstm_param.recurrent_to_forget_effective_bias,
            op_data_lstm.integer_lstm_param.input_to_cell_effective_bias,
            op_data_lstm.integer_lstm_param.recurrent_to_cell_effective_bias,
            op_data_lstm.integer_lstm_param.input_to_output_effective_bias,
            op_data_lstm.integer_lstm_param.recurrent_to_output_effective_bias,
            op_data_lstm.integer_lstm_param.input_to_input_effective_bias,
            op_data_lstm.integer_lstm_param.recurrent_to_input_effective_bias,
            /*n_batch=*/1, n_cell, n_input, n_output, output_state_ptr,
            op_data_lstm.inter_gate_parameters.output_mul_params.output_offset,
            cell_state_ptr, output_ptr, scratch0, scratch1, scratch2, scratch3,
            scratch4);
      }
    }
  }

  return kTfLiteOk;
}

}  // namespace lstm_xtensa

}  // namespace tflite

#endif  // defined(HIFI4) || defined(HIFI5)
