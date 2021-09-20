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

#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_pooling.h"

namespace tflite {
  TfLiteStatus AveragePoolingPrepareVision(TfLiteContext* context, TfLiteNode* node) {
    TFLITE_DCHECK(node->user_data != nullptr);
    TFLITE_DCHECK(node->builtin_data != nullptr);

    XtensaOpDataPooling* data = static_cast<XtensaOpDataPooling*>(node->user_data);
    const auto& params = *(static_cast<const TfLitePoolParams*>(node->builtin_data));

    const TfLiteTensor* input = GetInput(context, node, kPoolingInputTensor);
    TF_LITE_ENSURE(context, input != nullptr);
    TfLiteTensor* output = GetOutput(context, node, kPoolingOutputTensor);
    TF_LITE_ENSURE(context, output != nullptr);
    TFLITE_DCHECK_EQ(params.stride_height, params.stride_width);

    uint32_t inputN = input->dims->data[0];
    uint32_t inputH = input->dims->data[1];
    uint32_t inputW = input->dims->data[2];
    uint32_t inputD = input->dims->data[3];

    uint32_t outputN = output->dims->data[0];
    uint32_t outputH = output->dims->data[1];
    uint32_t outputW = output->dims->data[2];
    uint32_t outputD = output->dims->data[3];
    uint32_t strideWidth = params.stride_width;
    uint32_t filterWidth = params.filter_height;
    uint32_t filterHeight = params.filter_width;
    
    uint32_t contextSize = 0;
    uint32_t status = xiPoolGetMemReqd_Context(&contextSize);
    if (!status && contextSize) {
      void* data2 = context->AllocatePersistentBuffer(context, contextSize);
      if (data2 == nullptr) {
        return kTfLiteError;
      }
      data->pContext = (uint8_t*)data2;
      data->contextSize = contextSize;
    }

    status = xiAveragePoolSetContext(data->pContext, data->contextSize, inputD, inputW, inputH, inputN,
      outputD, outputW, outputH, outputN, filterWidth, filterHeight, 
      strideWidth, data->reference_op_data.padding.height, data->reference_op_data.padding.width,
      data->reference_op_data.activation_min, data->reference_op_data.activation_max);
    if (status) {
      return kTfLiteError;
    }

    return kTfLiteOk;
  }
  TfLiteStatus AveragePoolingEvalQuantizedVision(TfLiteContext* context, TfLiteNode* node) {
    const XtensaOpDataPooling* data =
      static_cast<const XtensaOpDataPooling*>(node->user_data);
    const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
    TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

    int8_t* input_data = (int8_t*)tflite::micro::GetTensorData<int8_t>(input);
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
    uint32_t input_size =
      input->dims->data[0] * input->dims->data[1] *
      input->dims->data[2] * input->dims->data[3];
    uint32_t output_size =
      output->dims->data[0] * output->dims->data[1] *
      output->dims->data[2] * output->dims->data[3];

    xiAverageEvalQuantized(data->pContext, data->contextSize, input_data, input_size, output_data, output_size);
    return kTfLiteOk;

  }

}  // namespace tflite

#endif //VISIONP6
