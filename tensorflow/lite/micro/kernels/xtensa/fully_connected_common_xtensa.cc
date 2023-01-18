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
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_fully_connected.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

void* XtensaInitFullyConnected(TfLiteContext* context, const char* buffer,
                               size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if !defined(VISION_P6)
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataFullyConnected));
#else
  void* data = context->AllocatePersistentBuffer(
      context, sizeof(XtensaFullyConnectedOpData));
  if (InitXtensaContext()) {
    return nullptr;
  }
  return data;
#endif  // defined(VISION_P6)
}

}  // namespace tflite
