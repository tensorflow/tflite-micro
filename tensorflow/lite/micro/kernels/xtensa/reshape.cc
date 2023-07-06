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

#include "tensorflow/lite/micro/kernels/reshape.h"

#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_reshape.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

#if defined(VISION_P6)
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data =
      context->AllocatePersistentBuffer(context, sizeof(XtensaReshapeData));
  if (InitXtensaContext()) {
    return nullptr;
  }
  return data;
}
#endif  // defined(VISION_P6)

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(PrepareReshapeReference(context, node));
#if defined(VISION_P6)
  {
    MicroContext* micro_context = GetMicroContext(context);
    TfLiteTensor* input =
        micro_context->AllocateTempInputTensor(node, kReshapeInputTensor);
    // Vision P6 currently only supports up to 4D int8 input tensors
    if (NumDimensions(input) <= 4 && input->type == kTfLiteInt8) {
      TF_LITE_ENSURE_OK(context, ReshapePrepareVision(context, node));
    }
    micro_context->DeallocateTempTfLiteTensor(input);
  }
#endif  // VISION_P6
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kReshapeInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kReshapeOutputTensor);

  // TODO(b/162522304): storing input bytes in OpData increases some models
  // significantly, possibly due to alignment issues.
  size_t input_bytes = EvalTensorBytes(input);

  // Do nothing for in-place reshape.
  if (input->data.raw != output->data.raw) {
    // Otherwise perform reshape with copy.
#if defined(VISION_P6)
    // Vision P6 currently only supports upto 4D int8 input tensors
    if (input->dims->size <= 4 && input->type == kTfLiteInt8) {
      XtensaReshapeData* op_data_xtensa =
          static_cast<XtensaReshapeData*>(node->user_data);
      ReshapeEvalVision(*op_data_xtensa, input, output);
    } else {
      memcpy(output->data.raw, input->data.raw, input_bytes);
    }
#else  // !defined(VISION_P6)
    memcpy(output->data.raw, input->data.raw, input_bytes);
#endif
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_RESHAPE() {
#if defined(VISION_P6)
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
#else
  return tflite::micro::RegisterOp(nullptr, Prepare, Eval);
#endif
}

}  // namespace tflite
