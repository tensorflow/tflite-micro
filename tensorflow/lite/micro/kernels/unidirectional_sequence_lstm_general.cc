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
#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval_general.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"

namespace tflite {

namespace {
/*Helper Functions*/

// Interface to access all the TempTfLiteTensors of the LSTM kernel
// Can only be constructed through the constructor to avoid memory leakage
// All TempTfLiteTensors will be deallocated through the destructor.
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

  // Verify the tensor properties
  // Input/output/states/FC weights tensors are required for kernel evaulation.
  // Also, the state tensors should be variables. Variants of the standard LSTM
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
    // cell state
    TF_LITE_ENSURE(context, internal_tensors_[kLstmCellStateTensor] != nullptr);
    TF_LITE_ENSURE(context,
                   internal_tensors_[kLstmCellStateTensor]->is_variable);
    // output
    TF_LITE_ENSURE(context, output_tensor_ != nullptr);

    // weight tensors (1-9, see lstm_shared for index definition)
    for (size_t i = 1; i < 9; i++) {
      TF_LITE_ENSURE(context, internal_tensors_[i] != nullptr);
    }
    // bias tensors (12-15, see lstm_shared for index definition)
    for (size_t i = 12; i < 16; i++) {
      TF_LITE_ENSURE(context, internal_tensors_[i] != nullptr);
    }

    // Tensors from LSTM variants are invalid
    // No peephole
    for (size_t i = 9; i < 12; i++) {
      TF_LITE_ENSURE(context, internal_tensors_[i] != nullptr);
    }
    // No projection
    for (size_t i = 16; i < 18; i++) {
      TF_LITE_ENSURE(context, internal_tensors_[i] != nullptr);
    }
    // No internal layer norm
    for (size_t i = 20; i < 24; i++) {
      TF_LITE_ENSURE(context, internal_tensors_[i] != nullptr);
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

void* UnidirectionalSequenceLstmInit(TfLiteContext* context, const char* buffer,
                                     size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataLSTM));
}

TfLiteStatus UnidirectionalSequenceLstmPrepare(TfLiteContext* context,
                                               TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 24);

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

  TF_LITE_ENSURE_OK(
      context, ValidateTensorSize(context, lstm_tensors, op_data->size_info));
}

TfLiteStatus UnidirectionalSequenceLstmEval(TfLiteContext* context,
                                            TfLiteNode* node) {}

}  // namespace

TfLiteRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM_General() {
  return tflite::micro::RegisterOp(UnidirectionalSequenceLstmInit,
                                   UnidirectionalSequenceLstmPrepare,
                                   UnidirectionalSequenceLstmEval);
}
}  // namespace tflite
