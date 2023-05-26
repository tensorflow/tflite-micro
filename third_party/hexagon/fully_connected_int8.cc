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

/* Copyright 2020 The Qualcomm Innovation Center, Inc. All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Qualcomm Innovation Center, Inc. nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
==============================================================================*/

#include "hexagon_tflm_translation_fully_connected.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "third_party/hexagon/hexagon_fully_connected.h"
#include "third_party/hexagon/hexagon_tflm_translation_fully_connected.h"

namespace tflite {
namespace {

TfLiteStatus EvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                               const HexagonOpDataFullyConnected& data,
                               const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = -data.reference_op_data.input_zero_point;
  op_params.weights_offset = -data.reference_op_data.filter_zero_point;
  op_params.output_offset = data.reference_op_data.output_zero_point;
  op_params.output_multiplier = data.reference_op_data.output_multiplier;
  // TODO(b/138810107): Figure out whether output shift should be inverted
  op_params.output_shift = data.reference_op_data.output_shift;
  op_params.quantized_activation_min =
      data.reference_op_data.output_activation_min;
  op_params.quantized_activation_max =
      data.reference_op_data.output_activation_max;

  const int32_t* bias_data =
      nullptr != bias ? tflite::micro::GetTensorData<int32_t>(bias) : nullptr;

  reference_integer_ops::FullyConnected(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias), bias_data,
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));

  return kTfLiteOk;
}

}  // namespace

void* HexagonFullyConnectedInit(TfLiteContext* context, const char* buffer,
                                size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = nullptr;
  data = context->AllocatePersistentBuffer(context,
                                           sizeof(HexagonOpDataFullyConnected));

  if (data == nullptr) {
    return nullptr;
  }
  HexagonOpDataFullyConnected* opdata =
      static_cast<HexagonOpDataFullyConnected*>(data);
  opdata->hexagon_data =
      tflite::hexagon_fully_connected::HexagonInit(context, buffer, length);

  return data;
}

TfLiteStatus HexagonFullyConnectedPrepare(TfLiteContext* context,
                                          TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  HexagonOpDataFullyConnected* data =
      static_cast<HexagonOpDataFullyConnected*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_OK(
      context, CalculateOpDataFullyConnected(context, params->activation,
                                             input->type, input, filter, bias,
                                             output, &data->reference_op_data));

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  tflite::hexagon_fully_connected::HexagonOptimizationEvaluation(context, node);

  if (tflite::hexagon_fully_connected::HexagonOptimizable(context, node)) {
    return tflite::hexagon_fully_connected::HexagonPrepare(context, node);
  }
  return kTfLiteOk;
}

TfLiteStatus HexagonFullyConnectedEvalInt8(TfLiteContext* context,
                                           TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const HexagonOpDataFullyConnected& data =
      *(static_cast<const HexagonOpDataFullyConnected*>(node->user_data));

  // This kernel only implements the int8 version of the fully_connected kernel.
  TFLITE_DCHECK(input->type == kTfLiteInt8);
  TFLITE_DCHECK(filter->type == kTfLiteInt8);
  if (bias != nullptr) {
    TFLITE_DCHECK(bias->type == kTfLiteInt32);
  }
  TFLITE_DCHECK(output->type == kTfLiteInt8);

  if (tflite::hexagon_fully_connected::HexagonOptimizable(context, node)) {
    return tflite::hexagon_fully_connected::HexagonEvalQuantizedInt8(
        context, node, node->user_data, input, filter, bias, output);
  } else {
    return EvalQuantizedInt8(context, node, data, input, filter, bias, output);
  }
  return kTfLiteOk;
}

TFLMRegistration Register_FULLY_CONNECTED_INT8() {
  return tflite::micro::RegisterOp(HexagonFullyConnectedInit,
                                   HexagonFullyConnectedPrepare,
                                   HexagonFullyConnectedEvalInt8);
}

}  // namespace tflite
