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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_depthwise_conv.h"

namespace tflite {

void* DepthwiseConvInitXtensa(TfLiteContext* context, const char* buffer,
                              size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = context->AllocatePersistentBuffer(
      context, sizeof(XtensaDepthwiseConvOpData));
#if defined(VISION_P6)
  if (InitXtensaContext()) {
    return nullptr;
  }
#endif  // defined(VISION_P6)

  return data;
}

TfLiteStatus DepthwiseConvPrepareXtensa(TfLiteContext* context,
                                        TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, DepthwiseConvPrepare(context, node));

#if defined(HIFI4) || defined(HIFI5) || defined(VISION_P6)
  auto op_data = reinterpret_cast<XtensaDepthwiseConvOpData*>(node->user_data);
  // optimized kernels can change this during Prepare
  op_data->can_optimize = false;
#endif  // defined(HIFI4) || defined(HIFI5) || defined(VISION_P6)

#if defined(HIFI4) || defined(HIFI5)
  TF_LITE_ENSURE_OK(context, DepthwiseConvPrepareHifi(context, node));
#endif  // defined(HIFI4) || defined(HIFI5)

#if defined(VISION_P6)
  TF_LITE_ENSURE_OK(context, DepthwiseConvPrepareVision(context, node));
#endif  // defined(VISION_P6)

  return kTfLiteOk;
}

}  // namespace tflite
