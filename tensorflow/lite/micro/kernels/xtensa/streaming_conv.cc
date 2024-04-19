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
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_conv.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);

  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  const auto& op_data =
      *(reinterpret_cast<XtensaStreamingConvOpData*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kConvBiasTensor);

  switch (input->type) {
    case kTfLiteInt16: {
      // Note that int32 bias is not widely supported and might be risky (e.g.
      // http://b/262003750). As such, while we have a fallback to the reference
      // implementation, production use-cases should only have int64 bias.
      if (bias->type == kTfLiteInt32) {
        // Streaming conv ref is not implemented for 32-bit bias
        MicroPrintf("Input Type %s (%d) with Bias Type %s (%d) not supported.",
                    TfLiteTypeGetName(input->type), input->type,
                    TfLiteTypeGetName(bias->type), bias->type);
        return kTfLiteError;
      } else {
        return StreamingConvEvalHifiInt16(context, node, params, op_data, input,
                                          filter, bias, output);
      }
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_STREAMING_CONV_2D() {
  return tflite::micro::RegisterOp(StreamingConvInitXtensa,
                                   StreamingConvPrepareXtensa, Eval);
}

}  // namespace tflite
