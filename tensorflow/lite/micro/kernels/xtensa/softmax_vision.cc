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
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_softmax.h"

namespace tflite {

TfLiteStatus SoftmaxPrepareVision(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  XtensaSoftmaxOpData* data =
      reinterpret_cast<XtensaSoftmaxOpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);

  uint32_t context_size = 0;
  uint32_t status = xiSoftmaxGetMemReqd_Context(&context_size);
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

  status = xiSoftmaxSetContext(
      data->p_context, data->context_size, input_dims[0], input_dims[1],
      input_dims[2], input_dims[3], data->params.input_multiplier,
      data->params.input_left_shift, data->params.diff_min);

  if (status) {
    return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);

  return kTfLiteOk;
}

TfLiteStatus SoftmaxEvalVision(TfLiteContext* context, TfLiteNode* node,
                               const XtensaSoftmaxOpData& data,
                               const TfLiteEvalTensor* input,
                               TfLiteEvalTensor* output) {
  const uint32_t input_size = NumElements(input->dims);
  const uint32_t output_size = NumElements(output->dims);

  xiSoftmax(data.p_context, data.context_size,
            const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input)),
            input_size, tflite::micro::GetTensorData<int8_t>(output),
            output_size);

  return kTfLiteOk;
}

}  // namespace tflite
#endif  // defined(VISION_P6)
