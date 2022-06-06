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

#if defined(VISION_P6)

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/add.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_add.h"

namespace tflite {

TfLiteStatus AddPrepareVision(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  XtensaAddOpData* data = reinterpret_cast<XtensaAddOpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kAddOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* input1 =
      micro_context->AllocateTempInputTensor(node, kAddInputTensor1);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* input2 =
      micro_context->AllocateTempInputTensor(node, kAddInputTensor2);
  TF_LITE_ENSURE(context, input2 != nullptr);

  uint32_t context_size = 0;
  uint32_t status = xiAddGetMemReqd_Context(&context_size);
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

  uint32_t input1_dims[4] = {1, 1, 1, 1};
  uint32_t input2_dims[4] = {1, 1, 1, 1};
  uint32_t output_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < NumDimensions(input1); i++) {
    input1_dims[i] =
        std::max(1, SizeOfDimension(input1, NumDimensions(input1) - 1 - i));
  }
  for (int i = 0; i < NumDimensions(input2); i++) {
    input2_dims[i] =
        std::max(1, SizeOfDimension(input2, NumDimensions(input2) - 1 - i));
  }
  for (int i = 0; i < NumDimensions(output); i++) {
    output_dims[i] =
        std::max(1, SizeOfDimension(output, NumDimensions(output) - 1 - i));
  }

  status = xiAddSetContext(
      data->p_context, data->context_size, input1_dims[0], input1_dims[1],
      input1_dims[2], input1_dims[3], input2_dims[0], input2_dims[1],
      input2_dims[2], input2_dims[3], output_dims[0], output_dims[1],
      output_dims[2], output_dims[3], input1->params.zero_point,
      input2->params.zero_point, output->params.zero_point,
      data->reference_op_data.input1_multiplier,
      data->reference_op_data.input2_multiplier,
      data->reference_op_data.output_multiplier,
      data->reference_op_data.input1_shift,
      data->reference_op_data.input2_shift,
      data->reference_op_data.output_shift,
      data->reference_op_data.output_activation_min,
      data->reference_op_data.output_activation_max);
  if (status) {
    return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input1);
  micro_context->DeallocateTempTfLiteTensor(input2);

  return kTfLiteOk;
}

TfLiteStatus AddEvalQuantizedVision(TfLiteContext* context, TfLiteNode* node,
                                    const TfLiteAddParams& params,
                                    const XtensaAddOpData& data,
                                    const TfLiteEvalTensor* input1,
                                    const TfLiteEvalTensor* input2,
                                    TfLiteEvalTensor* output) {
  const uint32_t input1_size = NumElements(input1->dims);
  const uint32_t input2_size = NumElements(input2->dims);
  const uint32_t output_size = NumElements(output->dims);

  xiAdd(data.p_context, data.context_size,
        const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input1)),
        input1_size,
        const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input2)),
        input2_size, tflite::micro::GetTensorData<int8_t>(output), output_size);
  return kTfLiteOk;
}
}  // namespace tflite
#endif  // defined(VISION_P6)
