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
#include "tensorflow/lite/micro/kernels/xtensa/lstm_eval.h"

#include <limits>

#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {

LstmTensors::LstmTensors(TfLiteContext* context, TfLiteNode* node) {
  micro_context_ = GetMicroContext(context);
  // 24 internal tensors. see lstm_shared.h for tensor names
  for (size_t i = 0; i < 24; i++) {
    internal_tensors_[i] = micro_context_->AllocateTempInputTensor(node, i);
  }
  output_tensor_ =
      micro_context_->AllocateTempOutputTensor(node, kLstmOutputTensor);
}

LstmTensors::~LstmTensors() {
  for (size_t i = 0; i < 24; i++) {
    if (internal_tensors_[i] != nullptr) {
      micro_context_->DeallocateTempTfLiteTensor(internal_tensors_[i]);
    }
  }
  micro_context_->DeallocateTempTfLiteTensor(output_tensor_);
}

// Verify the LSTM internal tensor properties (e.g., type checks)
// Input/output/states/fc weights tensors are required for kernel evaulation.
// The state tensors should be variables. Variants of the standard LSTM
// are not supported here, therefore their corresponding tensors should be
// invalid
TfLiteStatus LstmTensors::ValidateTensorStatus(TfLiteContext* context) const {
  // Verify certain tensor properties
  // input tensor
  TF_LITE_ENSURE(context, internal_tensors_[kLstmInputTensor] != nullptr);
  // hidden state
  TF_LITE_ENSURE(context, internal_tensors_[kLstmOutputStateTensor] != nullptr);
  TF_LITE_ENSURE(context,
                 internal_tensors_[kLstmOutputStateTensor]->is_variable);
  // hidden state becomes input so they must have the same type
  TF_LITE_ENSURE_EQ(context, internal_tensors_[kLstmOutputStateTensor]->type,
                    internal_tensors_[kLstmInputTensor]->type);
  // cell state
  TF_LITE_ENSURE(context, internal_tensors_[kLstmCellStateTensor] != nullptr);
  TF_LITE_ENSURE(context, internal_tensors_[kLstmCellStateTensor]->is_variable);
  // output
  TF_LITE_ENSURE(context, output_tensor_ != nullptr);
  // output type is the same as the input type (activations)
  TF_LITE_ENSURE_EQ(context, output_tensor_->type,
                    internal_tensors_[kLstmInputTensor]->type);

  // weight tensors (1-9, see lstm_shared for index definition)
  const auto weight_type =
      internal_tensors_[kLstmInputToForgetWeightsTensor]->type;
  for (size_t i = 1; i < 9; i++) {
    TF_LITE_ENSURE(context, internal_tensors_[i] != nullptr);
    TF_LITE_ENSURE_EQ(context, internal_tensors_[i]->type, weight_type);
  }

  // bias tensors (12-15, see lstm_shared for index definition)
  const auto bias_type = internal_tensors_[kLstmForgetGateBiasTensor]->type;
  for (size_t i = 12; i < 16; i++) {
    TF_LITE_ENSURE(context, internal_tensors_[i] != nullptr);
    TF_LITE_ENSURE_EQ(context, internal_tensors_[i]->type, bias_type);
  }
  // Tensors from LSTM variants are invalid
  // No peephole
  for (size_t i = 9; i < 12; i++) {
    TF_LITE_ENSURE(context, internal_tensors_[i] == nullptr);
  }
  // No projection
  for (size_t i = 16; i < 18; i++) {
    TF_LITE_ENSURE(context, internal_tensors_[i] == nullptr);
  }
  // No internal layer norm
  for (size_t i = 20; i < 24; i++) {
    TF_LITE_ENSURE(context, internal_tensors_[i] == nullptr);
  }
  return kTfLiteOk;
}

namespace lstm_internal {

#if !(defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
const int32_t kInt16Max = std::numeric_limits<int16_t>::max();
const int32_t kInt16Min = std::numeric_limits<int16_t>::min();
#endif

void AddElementWise(const int16_t* input_1, const int16_t* input_2, int n_batch,
                    int n_input, int16_t* output) {
#if !(defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      int32_t sum = input_1[index] + input_2[index];
      const int32_t sum_clamped = std::min(kInt16Max, std::max(kInt16Min, sum));
      output[index] = static_cast<int16_t>(sum_clamped);
    }
  }
#else
  xa_nn_elm_add_16x16_16(output, input_1, input_2, n_batch * n_input);
#endif
}

void AddElementWise(const float* input_1, const float* input_2, int n_batch,
                    int n_input, float* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      output[index] = input_1[index] + input_2[index];
    }
  }
}

#if !(defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
void Sigmoid(const RuntimeShape& data_shape, int16_t* data) {
  reference_integer_ops::Logistic(
      0 /*data->input_multiplier*/, 0 /*data->input_left_shift */,
      data_shape.FlatSize() /*NumElements(input->dims)*/,
      data /* tflite::micro::GetTensorData<int16_t>(input) */,
      data /*tflite::micro::GetTensorData<int16_t>(output) */);
}

void Sigmoid(const RuntimeShape& data_shape, float* data) {
  reference_ops::Logistic(data_shape, data, data_shape, data);
}

void Tanh(int32_t cell_state_scale_power, const RuntimeShape& input_data_shape,
          int16_t* input_data, const RuntimeShape& output_data_shape,
          int16_t* output_data) {
  int32_t tanh_input_left_shift = (15 + cell_state_scale_power) - 3;
  int32_t input_multiplier = 0;
  if (tanh_input_left_shift < 0) /* handling negative shift value */
  {
    tanh_input_left_shift = -tanh_input_left_shift;
    input_multiplier = 3;
  }
  reference_integer_ops::Tanh(input_multiplier, tanh_input_left_shift,
                              input_data_shape, input_data, output_data_shape,
                              output_data);
}

void Tanh(int32_t cell_state_scale_power, const RuntimeShape& input_data_shape,
          float* input_data, const RuntimeShape& output_data_shape,
          float* output_data) {
  reference_ops::Tanh(input_data_shape, input_data, output_data_shape,
                      output_data);
}

// Input and output have the same shape in LSTM
void Mul(const RuntimeShape& shape, const ArithmeticParams& params,
         const int16_t* input1_data, const int16_t* input2_data,
         int8_t* output_data) {
  return reference_integer_ops::MulElementwise(
      shape.FlatSize(), params, input1_data, input2_data, output_data);
}

// Input and output have the same shape in LSTM
void Mul(const RuntimeShape& shape, const ArithmeticParams& params,
         const int16_t* input1_data, const int16_t* input2_data,
         int16_t* output_data) {
  return reference_integer_ops::MulElementwise(
      shape.FlatSize(), params, input1_data, input2_data, output_data);
}

// Input and output have the same shape in LSTM
void Mul(const RuntimeShape& shape, const ArithmeticParams& params,
         const float* input1_data, const float* input2_data,
         float* output_data) {
  return reference_ops::Mul(params, shape, input1_data, shape, input2_data,
                            shape, output_data);
}

void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape, const int8_t* input_data,
                    const RuntimeShape& filter_shape, const int8_t* filter_data,
                    const RuntimeShape& bias_shape, const int32_t* bias_data,
                    const RuntimeShape& output_shape, int16_t* output_data) {
  return tflite::reference_integer_ops::FullyConnected(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data);
}

void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape, const int16_t* input_data,
                    const RuntimeShape& filter_shape, const int8_t* filter_data,
                    const RuntimeShape& bias_shape, const int64_t* bias_data,
                    const RuntimeShape& output_shape, int16_t* output_data) {
  return tflite::reference_integer_ops::FullyConnected(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data);
}

void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape, const float* input_data,
                    const RuntimeShape& filter_shape, const float* filter_data,
                    const RuntimeShape& bias_shape, const float* bias_data,
                    const RuntimeShape& output_shape, float* output_data) {
  return tflite::reference_ops::FullyConnected(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data);
}
#else  // #if !(defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
void Sigmoid(int16_t* data, int32_t data_size) {
  xa_nn_vec_sigmoid_sym16s_sym16s(data, data, 0, 0, data_size);
}

void Sigmoid(float* data, int32_t data_size) {
  int data_dims[2] = {1, data_size};
  RuntimeShape data_shape(2, reinterpret_cast<const int32_t*>(data_dims));
  reference_ops::Logistic(data_shape, data, data_shape, data);
}

void Tanh(int32_t cell_state_scale_power, int16_t* input_data,
          int16_t* output_data, int32_t data_size) {
  int32_t tanh_input_left_shift = (15 + cell_state_scale_power) - 3;
  int32_t input_multiplier = 0;
  if (tanh_input_left_shift < 0) /* handling negative shift value */
  {
    tanh_input_left_shift = -tanh_input_left_shift;
#if (defined(USE_HIFI_ACT_TIE) && \
     (defined(AE_TANH16X4X2) || defined(AE_TANH16X4)))
    input_multiplier = 1;
#else
    input_multiplier = 3;
#endif
  }
  xa_nn_vec_tanh_sym16s_sym16s(output_data, input_data, input_multiplier,
                               tanh_input_left_shift, data_size);
}

void Tanh(int32_t cell_state_scale_power, float* input_data, float* output_data,
          int32_t data_size) {
  int data_dims[2] = {1, data_size};
  RuntimeShape data_shape(2, reinterpret_cast<const int32_t*>(data_dims));
  reference_ops::Tanh(data_shape, input_data, data_shape, output_data);
}

// Input and output have the same shape in LSTM
void Mul(const ArithmeticParams& params, const int16_t* input1_data,
         const int16_t* input2_data, int8_t* output_data, int32_t data_size) {
  xa_nn_elm_mul_sym16sxsym16s_asym8s(
      output_data, params.output_offset, params.output_shift,
      params.output_multiplier, params.quantized_activation_min,
      params.quantized_activation_max, input1_data, input2_data, data_size);
}

// Input and output have the same shape in LSTM
void Mul(const ArithmeticParams& params, const int16_t* input1_data,
         const int16_t* input2_data, int16_t* output_data, int32_t data_size) {
  int dims_4D[4] = {1, 1, 1, data_size};
  xa_nn_elm_mul_broadcast_4D_sym16sxsym16s_sym16s(
      output_data, dims_4D, params.output_shift, params.output_multiplier,
      params.quantized_activation_min, params.quantized_activation_max,
      input1_data, dims_4D, input2_data, dims_4D);
  return;
}

// Input and output have the same shape in LSTM
void Mul(const ArithmeticParams& params, const float* input1_data,
         const float* input2_data, float* output_data, int32_t data_size) {
  int dims_2D[2] = {1, data_size};
  RuntimeShape data_shape(2, reinterpret_cast<const int32_t*>(dims_2D));
  return reference_ops::Mul(params, data_shape, input1_data, data_shape,
                            input2_data, data_shape, output_data);
}

void FullyConnected(const FullyConnectedParams& params,
                    const int8_t* input_data, const int8_t* filter_data,
                    const int32_t* bias_data, int16_t* output_data,
                    const int num_batches, const int output_depth,
                    const int accum_depth) {
#pragma loop_count min = 1
  for (int b = 0; b < num_batches; b++) {
    xa_nn_matXvec_out_stride_sym8sxasym8s_16(
        output_data + b * output_depth, filter_data,
        input_data + b * accum_depth, bias_data, output_depth, accum_depth,
        accum_depth, 1, params.input_offset, params.output_multiplier,
        params.output_shift);
  }
  return;
}

void FullyConnected(const FullyConnectedParams& params,
                    const int16_t* input_data, const int8_t* filter_data,
                    const int64_t* bias_data, int16_t* output_data,
                    const int num_batches, const int output_depth,
                    const int accum_depth) {
  xa_nn_matmul_sym8sxsym16s_sym16s(
      output_data, filter_data, input_data, bias_data, output_depth,
      accum_depth, accum_depth, num_batches, accum_depth, output_depth, 1,
      params.input_offset, params.output_multiplier, params.output_shift,
      params.output_offset);
  return;
}

void FullyConnected(const FullyConnectedParams& params, const float* input_data,
                    const float* filter_data, const float* bias_data,
                    float* output_data, const int num_batches,
                    const int output_depth, const int accum_depth) {
  int input_dims[2] = {num_batches, output_depth};
  RuntimeShape input_shape(2, reinterpret_cast<const int32_t*>(input_dims));
  RuntimeShape bias_shape(1, bias_data == NULL ? 0 : output_depth);
  int filter_dims[2] = {output_depth, accum_depth};
  RuntimeShape filter_shape(2, reinterpret_cast<const int32_t*>(filter_dims));
  int output_dims[2] = {num_batches, output_depth};
  RuntimeShape output_shape(2, reinterpret_cast<const int32_t*>(output_dims));
  return tflite::reference_ops::FullyConnected(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data);
}
#endif  // #if !(defined(HIFI3) || defined(HIFI4) || defined(HIFI5))

void Clipping(const int v_size, const CellStateInfo& cell_state_info,
              int16_t* vector) {
  for (int i = 0; i < v_size; i++) {
    vector[i] =
        std::max(std::min(cell_state_info.quantized_cell_clip, vector[i]),
                 static_cast<int16_t>(-cell_state_info.quantized_cell_clip));
  }
}

void Clipping(const int v_size, const CellStateInfo& cell_state_info,
              float* vector) {
  for (int i = 0; i < v_size; i++) {
    vector[i] = std::max(std::min(cell_state_info.cell_clip, vector[i]),
                         -cell_state_info.cell_clip);
  }
}

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
void UpdateLstmCell(const LstmStepManager& step_info,
                    TfLiteEvalTensor* cell_state,
                    // Gate outputs
                    int16_t* forget_gate_output,
                    const int16_t* input_gate_output,
                    const int16_t* cell_gate_output,
                    // Mul parameters
                    const ArithmeticParams& forget_cell_mul_params,
                    const ArithmeticParams& input_mul_params,
                    const CellStateInfo& cell_state_info, int16_t* buffer) {
  auto cell_state_shape = step_info.StateShape();
  // Check offset validity to avoid memory overflow
  TFLITE_DCHECK_LE(step_info.CellStateOffset() + cell_state_shape.FlatSize(),
                   tflite::micro::GetTensorShape(cell_state).FlatSize());

  // Multiplier is equivalent to 0.5 here so adding 1 to shifts
  xa_nn_lstm_cell_state_update_16(
      tflite::micro::GetTensorData<int16_t>(cell_state) +
          step_info.CellStateOffset(),
      forget_gate_output, cell_gate_output, input_gate_output,
      forget_cell_mul_params.output_shift - 1,
      input_mul_params.output_shift - 1, cell_state_info.quantized_cell_clip,
      cell_state_shape.FlatSize());
}

void UpdateLstmCell(const LstmStepManager& step_info,
                    TfLiteEvalTensor* cell_state,
                    // Gate outputs
                    float* forget_gate_output, const float* input_gate_output,
                    const float* cell_gate_output,
                    // Mul parameters
                    const ArithmeticParams& forget_cell_mul_params,
                    const ArithmeticParams& input_mul_params,
                    const CellStateInfo& cell_state_info, float* buffer) {
  // Check offset validity to avoid memory overflow
  TFLITE_DCHECK_LE(
      step_info.CellStateOffset() + step_info.StateShape().FlatSize(),
      tflite::micro::GetTensorShape(cell_state).FlatSize());

  auto cell_state_shape = step_info.StateShape();
  // Forget Gate x Cell State
  Mul(forget_cell_mul_params, forget_gate_output,
      tflite::micro::GetTensorData<float>(cell_state) +
          step_info.CellStateOffset(),
      tflite::micro::GetTensorData<float>(cell_state) +
          step_info.CellStateOffset(),
      cell_state_shape.FlatSize());
  // Input Gate x Cell Gate
  Mul(input_mul_params, input_gate_output, cell_gate_output, buffer,
      cell_state_shape.FlatSize());

  // Update the cell state
  AddElementWise(tflite::micro::GetTensorData<float>(cell_state) +
                     step_info.CellStateOffset(),
                 buffer,
                 /*n_batch=*/cell_state_shape.DimsData()[0],
                 /*n_state=*/cell_state_shape.DimsData()[1],
                 tflite::micro::GetTensorData<float>(cell_state) +
                     step_info.CellStateOffset());

  if (cell_state_info.cell_clip > 0) {
    Clipping(cell_state_shape.FlatSize(), cell_state_info,
             tflite::micro::GetTensorData<float>(cell_state) +
                 step_info.CellStateOffset());
  }
}
#endif  // #if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

// Increment the data offset so the sigle time step invocation call can access
// the corresponding input/output tensor data at the time step
void LstmStepManager::UpdateTime() {
  current_time_ += 1;
  TFLITE_DCHECK_LE(current_time_, size_info_.time_steps);
  // default as one batch per inference
  int input_step = size_info_.input_dimension;
  int output_step = size_info_.state_dimension;
  // time major: batch inference
  if (size_info_.time_major) {
    input_step = input_step * size_info_.batch_size;
    output_step = output_step * size_info_.batch_size;
  }

  input_offset_ += input_step;
  output_offset_ += output_step;
}

// Increment the data offset so the sigle time step invocation call can access
// the corresponding hidden/cell state tensor data at the time step (for single
// batch inference only)
void LstmStepManager::UpdateBatch() {
  current_batch_ += 1;
  TFLITE_DCHECK_LE(current_batch_, size_info_.batch_size);
  // batch inference for time major: no action needed
  if (size_info_.time_major) {
    return;
  }
  // otherwise: singe batch inference, go to the next batch
  hidden_state_offset_ += size_info_.state_dimension;
  cell_state_offset_ += size_info_.state_dimension;
}

// Input shape for each single time LSTM invocation.
// Multi-batch for time_major input
RuntimeShape LstmStepManager::InputShape() const {
  int batch_size = 1;
  if (size_info_.time_major) {
    batch_size = size_info_.batch_size;
  }
  const int dims[2] = {batch_size, size_info_.input_dimension};
  const int32_t* dims_data = reinterpret_cast<const int32_t*>(dims);
  return RuntimeShape(2, dims_data);
}

// State shape (both hidden and cell) for each single time LSTM invocation.
// Multi-batch for time_major input
RuntimeShape LstmStepManager::StateShape() const {
  int batch_size = 1;
  if (size_info_.time_major) {
    batch_size = size_info_.batch_size;
  }
  const int dims[2] = {batch_size, size_info_.state_dimension};
  const int32_t* dims_data = reinterpret_cast<const int32_t*>(dims);
  return RuntimeShape(2, dims_data);
}

}  // namespace lstm_internal
}  // namespace tflite
