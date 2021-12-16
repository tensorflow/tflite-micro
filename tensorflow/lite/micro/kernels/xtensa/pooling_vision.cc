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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_pooling.h"

namespace tflite {
TfLiteStatus AveragePrepareXtensa(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  XtensaOpDataPooling* data =
      static_cast<XtensaOpDataPooling*>(node->user_data);
  const auto& params =
      *(static_cast<const TfLitePoolParams*>(node->builtin_data));

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

  uint32_t context_size = 0;
  uint32_t status = xiPoolGetMemReqd_Context(&context_size);
  if (!status && context_size) {
    void* context_data =
        context->AllocatePersistentBuffer(context, context_size);
    if (context_data == nullptr) {
      return kTfLiteError;
    }
    data->p_context = reinterpret_cast<uint8_t*>(context_data);
    data->context_size = context_size;
  }

  status = xiAveragePoolSetContext(
      data->p_context, data->context_size, inputD, inputW, inputH, inputN,
      outputD, outputW, outputH, outputN, filterWidth, filterHeight,
      strideWidth, data->reference_op_data.padding.height,
      data->reference_op_data.padding.width,
      data->reference_op_data.activation_min,
      data->reference_op_data.activation_max);
  if (status) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus AverageEvalQuantizedXtensa(TfLiteContext* context,
                                        const TfLiteNode* node,
                                        const TfLitePoolParams* params,
                                        const XtensaOpDataPooling* data,
                                        const TfLiteEvalTensor* input,
                                        TfLiteEvalTensor* output) {
  int8_t* input_data =
      const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input));
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
  const uint32_t input_size = NumElements(input->dims);
  const uint32_t output_size = NumElements(output->dims);

  xiAverageEvalQuantized(data->p_context, data->context_size, input_data,
                         input_size, output_data, output_size);
  return kTfLiteOk;
}
}  // namespace tflite

#endif  // VISIONP6