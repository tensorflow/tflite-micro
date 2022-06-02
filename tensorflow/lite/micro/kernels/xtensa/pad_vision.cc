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
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_pad.h"

namespace tflite {

inline void OperandDims4D(uint32_t* dims, TfLiteTensor* opnd) {
  for (int i = NumDimensions(opnd) - 1, j = 0; i >= 0; i--, j++) {
    dims[j] = SizeOfDimension(opnd, i);
  }
  return;
}

TfLiteStatus PadPrepareVision(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  XtensaPadData* data = reinterpret_cast<XtensaPadData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, /*index=*/0);
  TfLiteTensor* paddings =
      micro_context->AllocateTempInputTensor(node, /*index=*/1);
  TfLiteTensor* constant_values =
      NumInputs(node) == 3
          ? micro_context->AllocateTempInputTensor(node, /*index=*/2)
          : nullptr;
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, /*index=*/0);

  uint32_t inputDims[4] = {1, 1, 1, 1};
  OperandDims4D(inputDims, input);

  const int32_t* paddings_data = GetTensorData<int32_t>(paddings);
  uint32_t inputRank = NumDimensions(input);

  uint32_t context_size = 0;
  uint32_t status = xiPadGetMemReqd_Context(&context_size);
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
  int8_t pad_value;
  if (constant_values == nullptr) {
    pad_value = static_cast<uint8_t>(data->reference_op_data.output_zero_point);
  } else {
    pad_value = *constant_values->data.int8;
  }

  status = xiPadSetContext(data->p_context, data->context_size, inputDims,
                           paddings_data, pad_value, inputRank);

  if (status) {
    return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(paddings);
  if (constant_values != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(constant_values);
  }
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}
TfLiteStatus PadEvalVision(const XtensaPadData& data,
                           const TfLiteEvalTensor* input,
                           TfLiteEvalTensor* output) {
  const uint32_t input_size = NumElements(input->dims);
  const uint32_t output_size = NumElements(output->dims);

  xiPad(data.p_context, data.context_size,
        const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input)),
        input_size, tflite::micro::GetTensorData<int8_t>(output), output_size);
  return kTfLiteOk;
}
}  // namespace tflite
#endif  // defined(VISION_P6)
