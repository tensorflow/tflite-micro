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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_LSTM_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_LSTM_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"

namespace tflite {

// Since LSTM includes multiple intermediate stages, introducing the internal
// namespace to expose them for testing
namespace lstm_xtensa {

#if defined(HIFI5) || defined(HIFI4)

// Pamameters for integer LSTM.
struct IntegerLstmParameter {
  // Pre-calculate bias + zero_point * weight.
  int32_t* input_to_forget_effective_bias = nullptr;
  int32_t* recurrent_to_forget_effective_bias = nullptr;
  int32_t* input_to_cell_effective_bias = nullptr;
  int32_t* recurrent_to_cell_effective_bias = nullptr;
  int32_t* input_to_output_effective_bias = nullptr;
  int32_t* recurrent_to_output_effective_bias = nullptr;
  int32_t* input_to_input_effective_bias = nullptr;
  int32_t* recurrent_to_input_effective_bias = nullptr;
};

#endif  // defined(HIFI5) || defined(HIFI4)

struct XtensaOpDataLstm : OpDataLSTM {
#if defined(HIFI5) || defined(HIFI4)
  int scratch_index_4; /* scratch buffer 5 */
  IntegerLstmParameter integer_lstm_param;
#endif  // defined(HIFI5) || defined(HIFI4)
};

#if defined(HIFI5) || defined(HIFI4)

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
    int16_t* scratch2, int16_t* scratch3, int8_t* scratch4);

void calc_cell_state_without_cifg(int16_t* cell_state,
                                  const int16_t* forget_gate,
                                  const int16_t* cell_gate,
                                  const int16_t* input_gate, int shift1,
                                  int shift2, int clip, int num_elms);

void xa_nn_elm_mul_16x16_asym8s(int8_t* output, const int16_t* input_1,
                                const int16_t* input_2, int32_t multiplier,
                                int32_t shift, int32_t zero_point,
                                int num_elms);

#endif  // defined(HIFI5) || defined(HIFI4)

}  // namespace lstm_xtensa

#if defined(HIFI5) || defined(HIFI4)

TfLiteStatus XtensaUnidirectionalSequenceLstmPrepareInt8x8_16(
    TfLiteContext* context, TfLiteNode* node, const LstmTensors&);

#endif  // defined(HIFI5) || defined(HIFI4)

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_LSTM_H_
