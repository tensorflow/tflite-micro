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

// Integer version of unidirectional sequence lstm. Only the standard LSTM
// (defined in the keras LSTM layer, e.g., no peephole etc.) is supported here.
// Currently used by the 16 bits activation case only

#include "tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.h"

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_lstm.h"

namespace tflite {

namespace {
/*Helper Functions*/

/*Kernel functions*/

#if defined(HIFI4) || defined(HIFI5)
TfLiteStatus PrepareInt8(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 24);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  TFLITE_DCHECK(node->user_data != nullptr);

  lstm_xtensa::XtensaOpDataLstm* op_data =
      static_cast<lstm_xtensa::XtensaOpDataLstm*>(node->user_data);
  const auto* builtin_data =
      static_cast<TfLiteUnidirectionalSequenceLSTMParams*>(node->builtin_data);

  // All TempTfLiteTensors will be deallocated through the destructor.
  LstmTensors lstm_tensors(context, node);
  TF_LITE_ENSURE_OK(context, lstm_tensors.ValidateTensorStatus(context));

  op_data->cell_gate_nonlinear_type = builtin_data->activation;
  op_data->size_info =
      CreateLstmSizeInfo(builtin_data->time_major,
                         lstm_tensors.GetInternalTensor(kLstmInputTensor)->dims,
                         lstm_tensors.HiddenStateTensor()->dims);
  TF_LITE_ENSURE_OK(
      context, ValidateTensorSize(context, lstm_tensors, op_data->size_info));

  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  TF_LITE_ENSURE(context, builtin_data->cell_clip >= 0);

  // Create cell state information and gate parameters (Fully Connected and Mul)
  TF_LITE_ENSURE_TYPES_EQ(
      context, lstm_tensors.GetInternalTensor(kLstmInputTensor)->type,
      kTfLiteInt8);
  TF_LITE_ENSURE_TYPES_EQ(
      context,
      lstm_tensors.GetInternalTensor(kLstmInputToInputWeightsTensor)->type,
      kTfLiteInt8);
  auto cell_state_type =
      lstm_tensors.GetInternalTensor(kLstmCellStateTensor)->type;
  TF_LITE_ENSURE_TYPES_EQ(context, cell_state_type, kTfLiteInt16);
  op_data->cell_state_info = CreateLstmCellStateInfo(
      lstm_tensors.CellStateTensor()->params.scale, builtin_data->cell_clip);
  TF_LITE_ENSURE_OK(
      context, PrepareGateParametersInteger(context, lstm_tensors, op_data));

  // request buffers (four buffers)
  for (size_t i = 0; i < 4; i++) {
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context,
                                   op_data->size_info.batch_size *
                                       op_data->size_info.state_dimension *
                                       TfLiteTypeGetSize(cell_state_type),
                                   &(op_data->buffer_indices[i])));
  }

  return XtensaUnidirectionalSequenceLstmPrepareInt8x8_16(context, node,
                                                          lstm_tensors);
}

TfLiteStatus EvalLstmInt8(TfLiteContext* context,
                          const lstm_xtensa::XtensaOpDataLstm& op_data,
                          LSTMKernelContents& kernel_content,
                          const LSTMBuffers<int16_t>& buffers) {
  return lstm_xtensa::EvalInteger8x8_16Lstm(
      kernel_content.GetInternalTensor(kLstmInputTensor),
      kernel_content.GetInternalTensor(kLstmInputToInputWeightsTensor),
      kernel_content.GetInternalTensor(kLstmInputToForgetWeightsTensor),
      kernel_content.GetInternalTensor(kLstmInputToCellWeightsTensor),
      kernel_content.GetInternalTensor(kLstmInputToOutputWeightsTensor),
      kernel_content.GetInternalTensor(kLstmRecurrentToInputWeightsTensor),
      kernel_content.GetInternalTensor(kLstmRecurrentToForgetWeightsTensor),
      kernel_content.GetInternalTensor(kLstmRecurrentToCellWeightsTensor),
      kernel_content.GetInternalTensor(kLstmRecurrentToOutputWeightsTensor),
      /* forward_sequence= */ true, op_data.size_info.time_major, op_data,
      kernel_content.HiddenStateTensor(), kernel_content.CellStateTensor(),
      kernel_content.OutputTensor(), buffers.buffer0, buffers.buffer1,
      buffers.buffer2, buffers.buffer3,
      reinterpret_cast<int8_t*>(
          context->GetScratchBuffer(context, op_data.scratch_index_4)));
}

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
  const lstm_xtensa::XtensaOpDataLstm& op_data =
      *static_cast<lstm_xtensa::XtensaOpDataLstm*>(node->user_data);
  auto kernel_content = CreateLSTMKernelContent(context, node);

  // 8(activation)x8(weight)->16(cell) LSTM with 32 bits bias
  LSTMBuffers<int16_t> buffers =
      CreateLSTMBuffers<int16_t>(context, op_data.buffer_indices);
  return EvalLstmInt8(context, op_data, kernel_content, buffers);
}
#endif  // defined(HIFI4) || defined(HIFI5)

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(
      context, sizeof(lstm_xtensa::XtensaOpDataLstm));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, UnidirectionalSequenceLstmPrepare(context, node));

#if defined(HIFI4) || defined(HIFI5)
  // All TempTfLiteTensors will be deallocated through the destructor.
  LstmTensors lstm_tensors(context, node);
  if (lstm_tensors.GetInternalTensor(kLstmInputTensor)->type == kTfLiteInt8 &&
      lstm_tensors.GetInternalTensor(kLstmInputToInputWeightsTensor)->type ==
          kTfLiteInt8 &&
      lstm_tensors.GetInternalTensor(kLstmCellStateTensor)->type ==
          kTfLiteInt16) {
    return XtensaUnidirectionalSequenceLstmPrepareInt8x8_16(context, node,
                                                            lstm_tensors);
  }
#endif  // defined(HIFI4) || defined(HIFI5)

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const lstm_xtensa::XtensaOpDataLstm& op_data =
      *static_cast<lstm_xtensa::XtensaOpDataLstm*>(node->user_data);
  auto kernel_content = CreateLSTMKernelContent(context, node);

  const auto activation_type =
      kernel_content.internal_tensors[kLstmInputTensor]->type;
  const auto weight_type =
      kernel_content.internal_tensors[kLstmInputToInputWeightsTensor]->type;

  switch (activation_type) {
    case kTfLiteFloat32: {
      LSTMBuffers<float> buffers =
          CreateLSTMBuffers<float>(context, op_data.buffer_indices);
      EvalLstm<float, float, float, float>(op_data, kernel_content, buffers);
      break;
    }
    case kTfLiteInt8: {
      switch (weight_type) {
        case kTfLiteInt8: {
          // 8(activation)x8(weight)->16(cell) LSTM with 32 bits bias
          LSTMBuffers<int16_t> buffers =
              CreateLSTMBuffers<int16_t>(context, op_data.buffer_indices);
#if defined(HIFI4) || defined(HIFI5)
          return EvalLstmInt8(context, op_data, kernel_content, buffers);
#else
          EvalLstm<int8_t, int8_t, int16_t, int32_t>(op_data, kernel_content,
                                                     buffers);
          break;
#endif  // defined(HIFI4) || defined(HIFI5)
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(weight_type), activation_type);
          return kTfLiteError;
        }
      }
      break;
    }
    case kTfLiteInt16: {
      switch (weight_type) {
        case kTfLiteInt8: {
          // 16(activation)x8(weight)->16(cell) LSTM with 64 bits bias
          LSTMBuffers<int16_t> buffers =
              CreateLSTMBuffers<int16_t>(context, op_data.buffer_indices);
          EvalLstm<int16_t, int8_t, int16_t, int64_t>(op_data, kernel_content,
                                                      buffers);
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(weight_type), weight_type);
          return kTfLiteError;
        }
      }
      break;
    }
    default: {
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(activation_type), activation_type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

#if defined(HIFI4) || defined(HIFI5)
TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT8() {
  return tflite::micro::RegisterOp(Init, PrepareInt8, EvalInt8);
}
#else
TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT8() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}
#endif  // defined(HIFI4) || defined(HIFI5)

}  // namespace tflite
