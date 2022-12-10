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
#include "tensorflow/lite/micro/kernels/lstm_eval_16act.h"

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace lstm_internal {



// // Calculates a single LSTM gate.
// // Implements the following formula:
// //   gate = activate(FC(input) + FC(recurrent))
// // Activation is sigmoid except for the "cell" gate (configurable, usually tanh)
// void CalculateLstmGateInteger(  // Input FC
//     const TfLiteEvalTensor* input, const TfLiteEvalTensor* input_weight,
//     const TfLiteEvalTensor* input_bias, const FullyConnectedParams * input_fc_params,
//     // Recurrent FC
//     const TfLiteEvalTensor* recurrent, const TfLiteEvalTensor* recurrent_weight,
//     const TfLiteEvalTensor* recurrent_bias, const FullyConnectedParams *  recurrent_fc_params,
//     // Output
//     CellType*  gate_output,
//     CellType* fc_output_buffer);
}  // namespace lstm_internal
}  // namespace tflite
