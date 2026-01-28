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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "pooling_rvv.h"

namespace tflite {

namespace {

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPooling* data =
      static_cast<const OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32:
      AveragePoolingEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteInt8:
      AveragePoolingEvalQuantized<int8_t>(context, node, params, data, input,
                                          output);
      break;
    case kTfLiteInt16:
      AveragePoolingEvalQuantized<int16_t>(context, node, params, data, input,
                                           output);
      break;
    default:
      MicroPrintf("Input type %s is not currently supported",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPooling* data =
      static_cast<const OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32:
      MaxPoolingEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteInt8:
    {
        tflite::PoolParams op_params;
        op_params.stride_height = params->stride_height;
        op_params.stride_width = params->stride_width;
        op_params.filter_height = params->filter_height;
        op_params.filter_width = params->filter_width;
        op_params.padding_values.height = data->padding.height;
        op_params.padding_values.width = data->padding.width;
        op_params.quantized_activation_min = data->activation_min;
        op_params.quantized_activation_max = data->activation_max;

        MaxPool8BitRVV(op_params,
                    tflite::micro::GetTensorShape(input),
                    tflite::micro::GetTensorData<std::int8_t>(input),
                    tflite::micro::GetTensorShape(output),
                    tflite::micro::GetTensorData<std::int8_t>(output));
    }
      break;
    case kTfLiteInt16:
    {
        tflite::PoolParams op_params;
        op_params.stride_height = params->stride_height;
        op_params.stride_width = params->stride_width;
        op_params.filter_height = params->filter_height;
        op_params.filter_width = params->filter_width;
        op_params.padding_values.height = data->padding.height;
        op_params.padding_values.width = data->padding.width;
        op_params.quantized_activation_min = data->activation_min;
        op_params.quantized_activation_max = data->activation_max;

        MaxPool16BitRVV(op_params,
                    tflite::micro::GetTensorShape(input),
                    tflite::micro::GetTensorData<std::int16_t>(input),
                    tflite::micro::GetTensorShape(output),
                    tflite::micro::GetTensorData<std::int16_t>(output));
    }
      break;
    default:
      MicroPrintf("Type %s not currently supported.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void* PoolInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataPooling));
}

}  // namespace

TFLMRegistration Register_AVERAGE_POOL_2D() {
  return tflite::micro::RegisterOp(PoolInit, PoolingPrepare, AverageEval);
}

TFLMRegistration Register_MAX_POOL_2D() {
  return tflite::micro::RegisterOp(PoolInit, PoolingPrepare, MaxEval);
}

}  // namespace tflite
