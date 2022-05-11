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

#if defined(VISION_P6)

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_fully_connected.h"

namespace tflite {

void NormalizeFCDims(uint32_t* dims, int rank) {
  if (rank < 4) {
    dims[3] = dims[rank - 1];
    dims[rank - 1] = 1;
  }
  dims[0] *= dims[1] * dims[2];
  dims[1] = 1;
  dims[2] = 1;
  return;
}

inline void OperandDims4D(uint32_t* dims, const TfLiteTensor* opnd) {
  for (int i = NumDimensions(opnd) - 1, j = 0; i >= 0; i--, j++) {
    dims[j] = SizeOfDimension(opnd, i);
  }
  return;
}
TfLiteStatus FullyConnectedPrepareVision(TfLiteContext* context,
                                         TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  XtensaFullyConnectedOpData* data =
      reinterpret_cast<XtensaFullyConnectedOpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);

  uint32_t inputDims[4] = {1, 1, 1, 1};
  uint32_t outputDims[4] = {1, 1, 1, 1};
  uint32_t filterDims[4] = {1, 1, 1, 1};

  OperandDims4D(inputDims, input);
  OperandDims4D(outputDims, output);
  OperandDims4D(filterDims, filter);

  NormalizeFCDims(inputDims, NumDimensions(input));
  NormalizeFCDims(filterDims, NumDimensions(filter));
  NormalizeFCDims(outputDims, NumDimensions(output));

  uint32_t context_size = 0;
  uint32_t status = xiFullyConnectedGetMemReqd_Context(&context_size);
  if (!status && context_size) {
    void* context_data =
        context->AllocatePersistentBuffer(context, context_size);
    if (context_data == nullptr) {
      return kTfLiteError;
    }
    data->p_context = reinterpret_cast<uint8_t*>(context_data);
    data->context_size = context_size;
  }

  status = xiFullyConnectedSetContext(
      data->p_context, data->context_size, inputDims, outputDims, filterDims, 1,
      input->params.zero_point, filter->params.zero_point,
      output->params.zero_point, data->reference_op_data.output_multiplier,
      data->reference_op_data.output_shift,
      data->reference_op_data.output_activation_min,
      data->reference_op_data.output_activation_max,
      (uint8_t*)GetTensorData<uint8_t>(filter));

  if (status) {
    return kTfLiteError;
  }

  uint32_t coefficient_size = 0;
  status = xiFullyConnectedGetMemReqd_Coeff(data->p_context, data->context_size,
                                            &coefficient_size);
  if (status || coefficient_size == 0) {
    return kTfLiteError;
  }

  void* coefficient_data =
      context->AllocatePersistentBuffer(context, coefficient_size);
  if (coefficient_data == nullptr) {
    return kTfLiteError;
  }
  data->reorder_coefficient_bias = reinterpret_cast<int8_t*>(coefficient_data);
  data->reorder_coefficient_bias_size = coefficient_size;

  status = xiFullyConnectedDoCoeffReorder(
      data->p_context, data->context_size,
      reinterpret_cast<uint8_t*>(data->reorder_coefficient_bias),
      data->reorder_coefficient_bias_size,
      const_cast<uint8_t*>(GetTensorData<uint8_t>(filter)),
      const_cast<int32_t*>(GetTensorData<int32_t>(bias)));
  if (status) {
    return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  return kTfLiteOk;
}

TfLiteStatus FullyConnectedEvalVision(TfLiteContext* context, TfLiteNode* node,
                                      const TfLiteConvParams& params,
                                      const XtensaFullyConnectedOpData& data,
                                      const TfLiteEvalTensor* input,
                                      const TfLiteEvalTensor* filter,
                                      const TfLiteEvalTensor* bias,
                                      TfLiteEvalTensor* output) {
  const uint32_t input_size = NumElements(input->dims);
  const uint32_t output_size = NumElements(output->dims);
  const int num_channels = filter->dims->data[kConvQuantizedDimension];

  xiFullyConnected(
      data.p_context, data.context_size,
      const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input)),
      input_size, tflite::micro::GetTensorData<int8_t>(output), output_size,
      data.reorder_coefficient_bias, data.reorder_coefficient_bias_size, NULL,
      NULL, num_channels);
  return kTfLiteOk;
}
}  // namespace tflite
#endif  // defined(VISION_P6)
