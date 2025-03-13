/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/maximum_minimum.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/maximum_minimum.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

template <KernelType kernel_type, typename OpType>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

  if (kernel_type == kReference) {
    switch (op_context.output->type) {
      case kTfLiteFloat32:
        TFLiteOperation<float, OpType>(context, node, op_context);
        break;
      case kTfLiteInt8:
        TFLiteOperation<int8_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt16:
        TFLiteOperation<int16_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt32:
        TFLiteOperation<int32_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt64:
        TFLiteOperation<int64_t, OpType>(context, node, op_context);
        break;
      default:
        MicroPrintf("Type %s (%d) is not supported by Maximum/Minimum.",
                    TfLiteTypeGetName(op_context.output->type),
                    op_context.output->type);
        return kTfLiteError;
    }
  } else {
    MicroPrintf("Kernel type not supported by Maximum/Minimum.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_MAXIMUM() {
  return tflite::micro::RegisterOp(nullptr, nullptr,
                                   Eval<kReference, MaximumOp>);
}

TFLMRegistration Register_MINIMUM() {
  return tflite::micro::RegisterOp(nullptr, nullptr,
                                   Eval<kReference, MinimumOp>);
}

}  // namespace tflite
