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

// Integer version of unidirectional sequence lstm. Only the standard LSTM
// (defined in the keras LSTM layer, e.g., no peephole etc.) is supported here.
// Currently used by the 16 bits activation case only

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"

namespace tflite {

namespace {
/*Helper Functions*/

// Interface to access all the TempTfLiteTensors of the LSTM kernel during the
// preparation phase. Can only be constructed through the constructor to avoid
// memory leakage. All TempTfLiteTensors will be deallocated through the
// destructor.
class LstmTensors {
 public:
  LstmTensors(const LstmTensors& other) = delete;
  LstmTensors& operator=(const LstmTensors& other) = delete;

  LstmTensors(TfLiteContext* context, TfLiteNode* node) {
    micro_context_ = GetMicroContext(context);
    // 24 internal tensors. see lstm_shared.h for tensor names
    for (size_t i = 0; i < 24; i++) {
      internal_tensors_[i] = micro_context_->AllocateTempInputTensor(node, i);
    }
    output_tensor_ =
        micro_context_->AllocateTempOutputTensor(node, kLstmOutputTensor);
  }

  ~LstmTensors() {
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
  TfLiteStatus ValidateTensorStatus(TfLiteContext* context) const {
    // Verify certain tensor properties
    // input tensor
    TF_LITE_ENSURE(context, internal_tensors_[kLstmInputTensor] != nullptr);
    // hidden state
    TF_LITE_ENSURE(context,
                   internal_tensors_[kLstmOutputStateTensor] != nullptr);
    TF_LITE_ENSURE(context,
                   internal_tensors_[kLstmOutputStateTensor]->is_variable);
    // hidden state becomes input so they must have the same type
    TF_LITE_ENSURE_EQ(context, internal_tensors_[kLstmOutputStateTensor]->type,
                      internal_tensors_[kLstmInputTensor]->type);
    // cell state
    TF_LITE_ENSURE(context, internal_tensors_[kLstmCellStateTensor] != nullptr);
    TF_LITE_ENSURE(context,
                   internal_tensors_[kLstmCellStateTensor]->is_variable);
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

  // Internal tensors. see lstm_shared.h for tensor names
  const TfLiteTensor* GetInternalTensor(const int tensor_index) const {
    return internal_tensors_[tensor_index];
  }

  const TfLiteTensor* HiddenStateTensor() const {
    return internal_tensors_[kLstmOutputStateTensor];
  }
  const TfLiteTensor* CellStateTensor() const {
    return internal_tensors_[kLstmCellStateTensor];
  }
  const TfLiteTensor* OutputTensor() const { return output_tensor_; }

 private:
  // see lstm_shared.h for tensor names
  MicroContext* micro_context_;
  TfLiteTensor* internal_tensors_[24];
  TfLiteTensor* output_tensor_;
};

// Deduce the size information (Batch (B), Time Steps (T), Input dimension (I),
// State dimension (S)) that defines the LSTM using the input and hidden state
// tensor
LstmSizeInfo CreateLstmSizeInfo(
    const bool time_major, const TfLiteIntArray* input_tensor_shape,
    const TfLiteIntArray* hidden_state_tensor_shape) {
  LstmSizeInfo size_info;
  size_info.time_major = time_major;
  size_info.batch_size =
      time_major ? input_tensor_shape->data[1] : input_tensor_shape->data[0];
  size_info.time_steps =
      time_major ? input_tensor_shape->data[0] : input_tensor_shape->data[1];
  size_info.input_dimension = input_tensor_shape->data[2];
  size_info.state_dimension = hidden_state_tensor_shape->data[1];
  return size_info;
}

TfLiteStatus ValidateWeightTensorSize(TfLiteContext* context,
                                      const TfLiteTensor* tensor, int dim1_size,
                                      int dim2_size) {
  TF_LITE_ENSURE_EQ(context, tensor->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, tensor->dims->data[0], dim1_size);
  TF_LITE_ENSURE_EQ(context, tensor->dims->data[1], dim2_size);
  return kTfLiteOk;
}

TfLiteStatus ValidateBiasTensorSize(TfLiteContext* context,
                                    const TfLiteTensor* tensor, int size) {
  TF_LITE_ENSURE_EQ(context, tensor->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, tensor->dims->data[0], size);
  return kTfLiteOk;
}

// Go through every tensors and make sure their shape match the kernel
// configuration
TfLiteStatus ValidateTensorSize(TfLiteContext* context,
                                const LstmTensors& tensors,
                                const LstmSizeInfo& size_info) {
  // Input FC weights
  for (size_t i = 1; i < 5; i++) {
    TF_LITE_ENSURE_OK(
        context, ValidateWeightTensorSize(context, tensors.GetInternalTensor(i),
                                          size_info.state_dimension,
                                          size_info.input_dimension));
  }
  // Recurrent FC weights
  for (size_t i = 5; i < 9; i++) {
    TF_LITE_ENSURE_OK(
        context, ValidateWeightTensorSize(context, tensors.GetInternalTensor(i),
                                          size_info.state_dimension,
                                          size_info.state_dimension));
  }
  // Biases
  for (size_t i = 12; i < 16; i++) {
    TF_LITE_ENSURE_OK(
        context, ValidateBiasTensorSize(context, tensors.GetInternalTensor(i),
                                        size_info.state_dimension));
  }

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  TF_LITE_ENSURE_EQ(context, NumElements(tensors.HiddenStateTensor()),
                    size_info.batch_size * size_info.state_dimension);
  TF_LITE_ENSURE_EQ(context, NumElements(tensors.CellStateTensor()),
                    size_info.batch_size * size_info.state_dimension);

  // Check the shape of output tensor against that of input tensor
  TF_LITE_ENSURE_EQ(context, tensors.OutputTensor()->dims->size, 3);
  TF_LITE_ENSURE_EQ(context,
                    tensors.GetInternalTensor(kLstmInputTensor)->dims->data[0],
                    tensors.OutputTensor()->dims->data[0]);
  TF_LITE_ENSURE_EQ(context,
                    tensors.GetInternalTensor(kLstmInputTensor)->dims->data[1],
                    tensors.OutputTensor()->dims->data[1]);
  TF_LITE_ENSURE_EQ(context, tensors.OutputTensor()->dims->data[2],
                    size_info.state_dimension);
  return kTfLiteOk;
}

// Wrapper function to create gate parameters for the four internal LSTM gates
TfLiteStatus CreateGateParams(
    TfLiteContext* context,
    /*Input tensors*/
    const TfLiteTensor* input, const TfLiteTensor* input_weight,
    const TfLiteTensor* input_bias,
    /*Hidden state tensors*/
    const TfLiteTensor* hidden_state, const TfLiteTensor* hidden_state_weight,
    const TfLiteTensor* hidden_state_bias,
    /*Scale of the fc output (input to non-linear activation)*/
    const float nonlinear_activation_input_scale, const TfLiteType cell_type,
    tflite::GateParameters& gate_params) {
  // A temp tflite tensor to represent the output of fc operation. Only the data
  // type and quantization parameters are set since it is only used for
  // parameter calculations
  TfLiteTensor fc_output_temp;
  fc_output_temp.type = cell_type;
  fc_output_temp.params.scale = nonlinear_activation_input_scale;
  fc_output_temp.params.zero_point = 0;  // symmetrical quantized

  // A temp fc opdata to reuse the helper function on creating fc parameters
  tflite::OpDataFullyConnected fc_data_temp;
  // TODO(http://b/265853320): due to the lack of precision for the float scale,
  // scale_diff / output_scale <= 0.02 (potentially requires 1e-8 precision) can
  // not be satisified for the bias. Here we rely on the correctiveness of the
  // conversion process (set input_bias=nullptr to avoid checking) for
  // tensor scales
  TF_LITE_ENSURE_STATUS(CalculateOpDataFullyConnected(
      context, kTfLiteActNone, input->type, input, input_weight,
      /*input_bias=*/nullptr, &fc_output_temp, &fc_data_temp));
  gate_params.input_fc_params = FullyConnectedParamsQuantized(fc_data_temp);
  double real_multiplier = 0.0;
  GetQuantizedConvolutionMultipler(context, input, input_weight, nullptr,
                                   &fc_output_temp, &real_multiplier);

  TF_LITE_ENSURE_STATUS(CalculateOpDataFullyConnected(
      context, kTfLiteActNone, hidden_state->type, hidden_state,
      hidden_state_weight, hidden_state_bias, &fc_output_temp, &fc_data_temp));
  gate_params.recurrent_fc_params = FullyConnectedParamsQuantized(fc_data_temp);
  return kTfLiteOk;
}

// Create parameters for element wise multiplication that happens in a) cell
// state update ; b) hidden state update
// Note that all the output of gates are symmetrically quantized so only scales
// are required for input. However, during the hidden state update phase, the
// output is the updated hidden state, which is asymmetrically quantized. Thus
// output may require zero point
tflite::ArithmeticParams CreateInterGateMulParams(const float input1_scale,
                                                  const float input2_scale,
                                                  const float output_scale,
                                                  const TfLiteType output_type,
                                                  const int output_zp = 0) {
  tflite::ArithmeticParams op_params = {};
  if (output_type == kTfLiteInt16) {
    op_params.quantized_activation_min = std::numeric_limits<int16_t>::min();
    op_params.quantized_activation_max = std::numeric_limits<int16_t>::max();
  } else if (output_type == kTfLiteInt8) {
    op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
    op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();
  }

  op_params.input1_offset = 0;  // symmetric
  op_params.input2_offset = 0;  // symmetric
  op_params.output_offset = output_zp;

  const double input_product_scale =
      static_cast<double>(input1_scale) * static_cast<double>(input2_scale);
  double effective_scale =
      input_product_scale / static_cast<double>(output_scale);

  QuantizeMultiplier(effective_scale, &op_params.output_multiplier,
                     &op_params.output_shift);
  return op_params;
}

// Create the additional information about the cell state, which include:
// cell_state_scale_power: used in integer nonlinear function (e.g., tanh)
// quantized_cell_clip: quantized cell clip range
CellStateInfo CreateLstmCellStateInfo(const float cell_state_scale,
                                      const float cell_clip) {
  CellStateInfo cell_state_info;
  // cell_state_scale_power: 2^-cell_state_scale_power = cell state scale
  int buffer;
  tflite::CheckedLog2(cell_state_scale, &buffer);
  cell_state_info.cell_state_scale_power = buffer;
  // Cell state specifics
  cell_state_info.cell_clip = cell_clip;
  cell_state_info.quantized_cell_clip = static_cast<int16_t>(
      std::min(std::max(static_cast<double>(cell_clip) /
                            static_cast<double>(cell_state_scale),
                        -32768.0),
               32767.0));
  return cell_state_info;
}

tflite::FullyConnectedParams CreateFCParamsFloat() {
  FullyConnectedParams op_params;
  CalculateActivationRange(kTfLiteActNone, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  return op_params;
}

tflite::GateParameters CreateGateParamsFloat() {
  tflite::GateParameters gate_params = {};
  gate_params.input_fc_params = CreateFCParamsFloat();
  gate_params.recurrent_fc_params = CreateFCParamsFloat();
  return gate_params;
}

tflite::ArithmeticParams CreateInterGateMulParamsFloat() {
  tflite::ArithmeticParams op_params = {};
  CalculateActivationRange(kTfLiteActNone, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  return op_params;
}

TfLiteStatus PrepareGateParametersFloat(TfLiteContext* context,
                                        const LstmTensors& lstm_tensors,
                                        OpDataLSTM* op_data) {
  // Gate Parameters
  op_data->forget_gate_parameters = CreateGateParamsFloat();
  op_data->input_gate_parameters = CreateGateParamsFloat();
  op_data->cell_gate_parameters = CreateGateParamsFloat();
  op_data->output_gate_parameters = CreateGateParamsFloat();
  // Inter gate multiplication parameters
  op_data->inter_gate_parameters.forget_cell_mul_params =
      CreateInterGateMulParamsFloat();
  op_data->inter_gate_parameters.input_mul_params =
      CreateInterGateMulParamsFloat();
  op_data->inter_gate_parameters.output_mul_params =
      CreateInterGateMulParamsFloat();
  return kTfLiteOk;
}

TfLiteStatus PrepareGateParametersInteger(TfLiteContext* context,
                                          const LstmTensors& lstm_tensors,
                                          OpDataLSTM* op_data) {
  float nonlinear_input_scale = 0.00024414062;  // 2^-12 Q3.12 -> Q0.15
  TF_LITE_ENSURE_OK(
      context,
      CreateGateParams(
          context, lstm_tensors.GetInternalTensor(kLstmInputTensor),
          lstm_tensors.GetInternalTensor(kLstmInputToForgetWeightsTensor),
          lstm_tensors.GetInternalTensor(kLstmForgetGateBiasTensor),
          lstm_tensors.GetInternalTensor(kLstmOutputStateTensor),
          lstm_tensors.GetInternalTensor(kLstmRecurrentToForgetWeightsTensor),
          /*hidden_state_bias=*/nullptr, nonlinear_input_scale, kTfLiteInt16,
          op_data->forget_gate_parameters));
  TF_LITE_ENSURE_OK(
      context,
      CreateGateParams(
          context, lstm_tensors.GetInternalTensor(kLstmInputTensor),
          lstm_tensors.GetInternalTensor(kLstmInputToInputWeightsTensor),
          lstm_tensors.GetInternalTensor(kLstmInputGateBiasTensor),
          lstm_tensors.GetInternalTensor(kLstmOutputStateTensor),
          lstm_tensors.GetInternalTensor(kLstmRecurrentToInputWeightsTensor),
          /*hidden_state_bias=*/nullptr, nonlinear_input_scale, kTfLiteInt16,
          op_data->input_gate_parameters));
  TF_LITE_ENSURE_OK(
      context,
      CreateGateParams(
          context, lstm_tensors.GetInternalTensor(kLstmInputTensor),
          lstm_tensors.GetInternalTensor(kLstmInputToCellWeightsTensor),
          lstm_tensors.GetInternalTensor(kLstmCellGateBiasTensor),
          lstm_tensors.GetInternalTensor(kLstmOutputStateTensor),
          lstm_tensors.GetInternalTensor(kLstmRecurrentToCellWeightsTensor),
          /*hidden_state_bias=*/nullptr, nonlinear_input_scale, kTfLiteInt16,
          op_data->cell_gate_parameters));
  TF_LITE_ENSURE_OK(
      context,
      CreateGateParams(
          context, lstm_tensors.GetInternalTensor(kLstmInputTensor),
          lstm_tensors.GetInternalTensor(kLstmInputToOutputWeightsTensor),
          lstm_tensors.GetInternalTensor(kLstmOutputGateBiasTensor),
          lstm_tensors.GetInternalTensor(kLstmOutputStateTensor),
          lstm_tensors.GetInternalTensor(kLstmRecurrentToOutputWeightsTensor),
          /*hidden_state_bias=*/nullptr, nonlinear_input_scale, kTfLiteInt16,
          op_data->output_gate_parameters));

  // Inter gate multiplication parameters
  float nonlinear_output_scale = 0.00003051757;  // 2^-15 Q3.12 -> Q0.15
  float cell_state_scale = lstm_tensors.CellStateTensor()->params.scale;
  // forget gate output (nonlinear output) x cell state -> cell state
  op_data->inter_gate_parameters.forget_cell_mul_params =
      CreateInterGateMulParams(nonlinear_output_scale, cell_state_scale,
                               cell_state_scale, kTfLiteInt16);
  // input gate output x cell gate output -> cell state
  op_data->inter_gate_parameters.input_mul_params =
      CreateInterGateMulParams(nonlinear_output_scale, nonlinear_output_scale,
                               cell_state_scale, kTfLiteInt16);
  // tanh output x output gate output -> hidden state (potentially asymmetric)
  op_data->inter_gate_parameters.output_mul_params = CreateInterGateMulParams(
      nonlinear_output_scale, nonlinear_output_scale,
      lstm_tensors.HiddenStateTensor()->params.scale,
      lstm_tensors.HiddenStateTensor()->type,
      lstm_tensors.HiddenStateTensor()->params.zero_point);
  return kTfLiteOk;
}

LSTMKernelContents CreateLSTMKernelContent(TfLiteContext* context,
                                           TfLiteNode* node) {
  LSTMKernelContents kernel_content;
  // Point to correct tensors
  for (size_t i = 0; i < 24; i++) {
    kernel_content.internal_tensors[i] =
        tflite::micro::GetMutableEvalInput(context, node, i);
  }
  // Output tensor
  kernel_content.output_tensor = tflite::micro::GetEvalOutput(context, node, 0);
  return kernel_content;
}

template <typename CellType>
LSTMBuffers<CellType> CreateLSTMBuffers(TfLiteContext* context,
                                        const int* buffer_indices) {
  LSTMBuffers<CellType> buffers;
  buffers.buffer0 = reinterpret_cast<CellType*>(
      context->GetScratchBuffer(context, buffer_indices[0]));
  buffers.buffer1 = reinterpret_cast<CellType*>(
      context->GetScratchBuffer(context, buffer_indices[1]));
  buffers.buffer2 = reinterpret_cast<CellType*>(
      context->GetScratchBuffer(context, buffer_indices[2]));
  buffers.buffer3 = reinterpret_cast<CellType*>(
      context->GetScratchBuffer(context, buffer_indices[3]));
  return buffers;
}

/*Kernel functions*/

void* UnidirectionalSequenceLstmInit(TfLiteContext* context, const char* buffer,
                                     size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataLSTM));
}

TfLiteStatus UnidirectionalSequenceLstmPrepare(TfLiteContext* context,
                                               TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 24);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  TFLITE_DCHECK(node->user_data != nullptr);

  OpDataLSTM* op_data = reinterpret_cast<OpDataLSTM*>(node->user_data);
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

  op_data->cell_state_info = CreateLstmCellStateInfo(
      lstm_tensors.CellStateTensor()->params.scale, builtin_data->cell_clip);
  TF_LITE_ENSURE_OK(
      context, ValidateTensorSize(context, lstm_tensors, op_data->size_info));

  // Create Gate Parameters (Fully Connected and Mul)
  auto cell_state_type =
      lstm_tensors.GetInternalTensor(kLstmCellStateTensor)->type;
  if (cell_state_type == kTfLiteFloat32) {
    MicroPrintf("Prepared Float!");
    TF_LITE_ENSURE_OK(
        context, PrepareGateParametersFloat(context, lstm_tensors, op_data));
  } else if (cell_state_type == kTfLiteInt16) {
    TF_LITE_ENSURE_OK(
        context, PrepareGateParametersInteger(context, lstm_tensors, op_data));
  } else {
    MicroPrintf(
        "Cell state type %s (%d) not supported. The quantized Unidirectional "
        "Sequence LSTM Op only support int16 cell state",
        TfLiteTypeGetName(cell_state_type), cell_state_type);
    return kTfLiteError;
  }
  // request buffers (four buffers)
  for (size_t i = 0; i < 4; i++) {
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context,
                                   op_data->size_info.batch_size *
                                       op_data->size_info.state_dimension *
                                       TfLiteTypeGetSize(cell_state_type),
                                   &(op_data->buffer_indices[i])));
  }
  return kTfLiteOk;
}

TfLiteStatus UnidirectionalSequenceLstmEval(TfLiteContext* context,
                                            TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataLSTM& op_data = *reinterpret_cast<OpDataLSTM*>(node->user_data);
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
          EvalLstm<int8_t, int8_t, int16_t, int32_t>(op_data, kernel_content,
                                                     buffers);
          break;
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

TfLiteRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM() {
  return tflite::micro::RegisterOp(UnidirectionalSequenceLstmInit,
                                   UnidirectionalSequenceLstmPrepare,
                                   UnidirectionalSequenceLstmEval);
}
}  // namespace tflite
