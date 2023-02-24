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
#if defined(VISION_P6)

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_pooling.h"

#define MAX_POOLING 0
#define AVG_POOLING 1

namespace tflite {

TfLiteStatus PoolingPrepareVision(TfLiteContext* context, TfLiteNode* node,
                                  uint8_t pool_type) {
  TF_LITE_ENSURE_STATUS(PoolingPrepare(context, node));
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  XtensaOpDataPooling* data =
      reinterpret_cast<XtensaOpDataPooling*>(node->user_data);
  const auto& params =
      *(reinterpret_cast<const TfLitePoolParams*>(node->builtin_data));

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kPoolingOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kPoolingInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);

  if (input->type == kTfLiteInt8) {
    uint32_t context_size = 0;
    uint32_t status = xiPoolGetMemReqd_Context(&context_size);
    TFLITE_DCHECK(status == 0);
    if (context_size) {
      void* context_data =
          context->AllocatePersistentBuffer(context, context_size);
      if (context_data == nullptr) {
        return kTfLiteError;
      }
      data->p_context = reinterpret_cast<uint8_t*>(context_data);
      data->context_size = context_size;
    }

    uint32_t input_dims[4] = {1, 1, 1, 1};
    uint32_t output_dims[4] = {1, 1, 1, 1};
    for (int i = 0; i < NumDimensions(input); i++) {
      input_dims[i] =
          std::max(1, SizeOfDimension(input, NumDimensions(input) - 1 - i));
    }
    for (int i = 0; i < NumDimensions(output); i++) {
      output_dims[i] =
          std::max(1, SizeOfDimension(output, NumDimensions(output) - 1 - i));
    }

    status = xiPoolSetContext(
        data->p_context, data->context_size, input_dims[0], input_dims[1],
        input_dims[2], input_dims[3], output_dims[0], output_dims[1],
        output_dims[2], params.filter_width, params.filter_height,
        params.stride_width, params.stride_height,
        data->reference_op_data.padding.width,
        data->reference_op_data.padding.height, input->params.zero_point,
        output->params.zero_point, data->reference_op_data.activation_min,
        data->reference_op_data.activation_max, pool_type);
    if (status) {
      return kTfLiteError;
    }
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);

  return kTfLiteOk;
}

TfLiteStatus AvgPoolingPrepareVision(TfLiteContext* context, TfLiteNode* node) {
  return PoolingPrepareVision(context, node, AVG_POOLING);
}

TfLiteStatus MaxPoolingPrepareVision(TfLiteContext* context, TfLiteNode* node) {
  return PoolingPrepareVision(context, node, MAX_POOLING);
}

TfLiteStatus PoolEvalVision(TfLiteContext* context, TfLiteNode* node,
                            const TfLitePoolParams& params,
                            const XtensaOpDataPooling& data,
                            const TfLiteEvalTensor* input,
                            TfLiteEvalTensor* output) {
  const uint32_t input_size = NumElements(input->dims);
  const uint32_t output_size = NumElements(output->dims);

  xiPool(data.p_context, data.context_size,
         const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input)),
         input_size, tflite::micro::GetTensorData<int8_t>(output), output_size);
  return kTfLiteOk;
}
}  // namespace tflite

#endif  // VISIONP6
