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

#if defined(VISIONP6)

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_depthwise_conv.h"

namespace tflite {

  TfLiteStatus DepthwiseConvPrepareVision(TfLiteContext* context, TfLiteNode* node) {
    TFLITE_DCHECK(node->user_data != nullptr);
    TFLITE_DCHECK(node->builtin_data != nullptr);

    XtensaDepthwiseConvOpData* data = static_cast<XtensaDepthwiseConvOpData*>(node->user_data);
    const auto& params =
      *(static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data));

    TfLiteTensor* output = GetOutput(context, node, kDepthwiseConvOutputTensor);
    TF_LITE_ENSURE(context, output != nullptr);
    const TfLiteTensor* input =
      GetInput(context, node, kDepthwiseConvInputTensor);
    TF_LITE_ENSURE(context, input != nullptr);
    const TfLiteTensor* filter =
      GetInput(context, node, kDepthwiseConvWeightsTensor);
    TF_LITE_ENSURE(context, filter != nullptr);
    const TfLiteTensor* bias =
      GetInput(context, node, kDepthwiseConvBiasTensor);
    TF_LITE_ENSURE(context, filter != nullptr);

   // const int input_width = input->dims->data[2];
   // const int input_height = input->dims->data[1];
    const int filter_width = filter->dims->data[2];
    const int filter_height = filter->dims->data[1];
   // const int output_width = output->dims->data[2];
   // const int output_height = output->dims->data[1];

    // Dynamically allocate per-channel quantization parameters.
    const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
    data->per_channel_output_shift_int8 =
      static_cast<int8_t*>(context->AllocatePersistentBuffer(
        context, num_channels));


    for (int i = 0; i < num_channels; i++)
    {
      data->per_channel_output_shift_int8[i] = (int8_t)(-1 * data->reference_op_data.per_channel_output_shift[i]);
    }

    data->enableXtensaKernel = 0;
    if ((input->type == kTfLiteInt8) &&
      (params.stride_width == params.stride_height) &&
      filter_width == 3 && filter_height == 3)
      data->enableXtensaKernel = 1;

    if (data->enableXtensaKernel) {
      uint32_t contextSize = 0;
      uint32_t status = xiDepthwiseConvGetMemReqd_Context(&contextSize);
      if (!status && contextSize) {
        void* data2 = context->AllocatePersistentBuffer(context, contextSize);
        if (data2 == nullptr) {
          return kTfLiteError;
        }
        data->pContext = (uint8_t*)data2;
        data->contextSize = contextSize;
      }

     // uint32_t inputN = SizeOfDimension(input, 0);
      uint32_t inputH = SizeOfDimension(input, 1);
      uint32_t inputW = SizeOfDimension(input, 2);
      uint32_t inputD = SizeOfDimension(input, 3);

     // uint32_t outputN = SizeOfDimension(output, 0);
      uint32_t outputH = SizeOfDimension(output, 1);
      uint32_t outputW = SizeOfDimension(output, 2);
      uint32_t outputD = SizeOfDimension(output, 3);

      uint32_t filterH = SizeOfDimension(filter, 1);
      uint32_t filterW = SizeOfDimension(filter, 2);

      status = xiDepthwiseConvSetContext(data->pContext, data->contextSize,
        inputD, inputW, inputH, outputD, outputW, outputH, filterW, filterH,
        params.stride_width, input->params.zero_point,
        filter->params.zero_point, output->params.zero_point,
        data->reference_op_data.output_multiplier, data->reference_op_data.output_shift,
        data->reference_op_data.output_activation_min, data->reference_op_data.output_activation_max);
      if (status)
        return kTfLiteError;

      uint32_t coeffSize = 0;
      status = xiDepthwiseConvGetMemReqd_Coeff(data->pContext, data->contextSize, &coeffSize);
      if (!status && coeffSize) {
        void* data2 = context->AllocatePersistentBuffer(context, coeffSize);
        if (data2 == nullptr) {
          return kTfLiteError;
        }
        data->reordCoeffnBias = (int8_t*)data2;
        data->reordCoeffnBiasSize = coeffSize;
      }
      else
        return kTfLiteError;

      status = xiDepthwiseConvDoCoeffReorder(data->pContext, data->contextSize,
        (uint8_t*)data->reordCoeffnBias, data->reordCoeffnBiasSize,
        (uint8_t*)GetTensorData<uint8_t>(filter),
        (int32_t*)GetTensorData<int32_t>(bias));
      if (status)
        return kTfLiteError;
    }

    return kTfLiteOk;
  }

  TfLiteStatus DepthwiseConvEvalVision(TfLiteContext* context, TfLiteNode* node) {
    auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
    const XtensaDepthwiseConvOpData& data = *(static_cast<const XtensaDepthwiseConvOpData*>(node->user_data));

    TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kDepthwiseConvOutputTensor);
    const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvInputTensor);
    const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvWeightsTensor);
    const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
      ? tflite::micro::GetEvalInput(context, node, kDepthwiseConvBiasTensor)
      : nullptr;

    if (data.enableXtensaKernel) {
      uint32_t input_size =
        input->dims->data[0] * input->dims->data[1] *
        input->dims->data[2] * input->dims->data[3];
      uint32_t output_size =
        output->dims->data[0] * output->dims->data[1] *
        output->dims->data[2] * output->dims->data[3];
      uint32_t num_channels =
        filter->dims->data[kDepthwiseConvQuantizedDimension];
      xiDepthwiseConv(data.pContext, data.contextSize,
        (int8_t*)tflite::micro::GetTensorData<int8_t>(input), input_size,
        tflite::micro::GetTensorData<int8_t>(output), output_size,
        data.reordCoeffnBias, data.reordCoeffnBiasSize,
        data.reference_op_data.per_channel_output_multiplier,
        data.per_channel_output_shift_int8, num_channels,
        data.reference_op_data.padding.width, data.reference_op_data.padding.height);
    }
    else {
        reference_integer_ops::DepthwiseConvPerChannel(
        DepthwiseConvParamsQuantized(params, data.reference_op_data),
          data.reference_op_data.per_channel_output_multiplier,
          data.reference_op_data.per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
    }
    return kTfLiteOk;
  }
}  // namespace tflite
#endif  // defined(VISIONP6)
