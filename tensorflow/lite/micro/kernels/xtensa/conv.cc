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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
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
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;

  switch (input->type) {
    case kTfLiteFloat32: {
#if defined(INCLUDE_FLOAT_OPT)
      return ConvEvalHifiFloat32(context, node, params, op_data, input, filter,
                   bias, output);
#else    
      return ConvReferenceEvalFloat32(context, node);
#endif          
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
#if defined(HIFI5) && defined(NNLIB_HIFI5)
          return ConvEvalHifiInt4(context, node, params, op_data, input, filter,
                       bias, output);
#elif defined(HIFI4)
          TfLiteEvalTensor filter_int8 = tflite::micro::MakeUnpackedInt4Tensor(
              context, op_data.reference_op_data.filter_buffer_index, filter);
          return ConvEvalHifiInt8(context, node, params, op_data, input, &filter_int8,
                           bias, output);
#else
          return ConvReferenceEvalInt8(context, node);
#endif        
          break;  
        } 
        case kTfLiteInt8: {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
          return ConvEvalHifiInt8(context, node, params, op_data, input, filter,
                           bias, output);
#elif defined(VISION_P6)
          // At this time the optimized implementation is failing the unit tests in
          // ways that are not entirely clear why. For now, we have identified some
          // of the problem cases and are manually inserting a reference fallback.
          // See http://b/270720625 for more details.
          if (op_data.is_per_channel_quantized ||
              input->dims->data[1] != input->dims->data[2]) {
            return ConvReferenceEvalInt8(context, node);
          } else {
            return ConvEvalVision(context, node, params, op_data, input, filter,
                                  bias, output);
          }
#else
          return ConvReferenceEvalInt8(context, node);
#endif
          break;
        }
        default:
          MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(filter->type),
                      filter->type);
          return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt16: {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      if (bias == nullptr || bias->type == kTfLiteInt64) {
        return ConvEvalHifiInt16(context, node, params, op_data, input, filter, bias,
                          output);
      }
      else if (bias->type == kTfLiteInt32) {
#else  // defined(HIFI4) || defined(HIFI5)
      if (bias == nullptr || bias->type == kTfLiteInt64 || bias->type == kTfLiteInt32) {
#endif  // defined(HIFI4) || defined(HIFI5)
        return ConvReferenceEvalInt16(context, node);
      }
      else {
        MicroPrintf("Bias type %s (%d) not supported.",
                    TfLiteTypeGetName(bias->type), bias->type);
        return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(ConvInitXtensa, ConvPrepareXtensa, Eval);
}

}  // namespace tflite
