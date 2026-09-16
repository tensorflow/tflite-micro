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
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_conv.h"

namespace tflite {
namespace {

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFIMINI)
  return ConvReferenceEvalInt8(context, node);
#else
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kConvBiasTensor);

  switch (filter->type) {
    case kTfLiteInt4: {
    #if defined(HIFI5) && defined(NNLIB_HIFI5) || defined(HIFI_IQ)
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
    }
    case kTfLiteInt8: {
    #if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
        return ConvEvalHifiInt8(context, node, params, op_data, input, filter, bias,
                        output);
    #else
        return ConvReferenceEvalInt8(context, node);                    
    #endif
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(filter->type),
                  filter->type);
      return kTfLiteError;     
  }
#endif  // defined(HIFIMINI)
}

TfLiteStatus EvalInt16(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kConvBiasTensor);

  if(bias == nullptr || bias->type == kTfLiteInt64){
    return ConvEvalHifiInt16(context, node, params, op_data, input, filter, bias,
                           output);
  }
  else if(bias->type == kTfLiteInt32){
    return ConvReferenceEvalInt16(context, node);
  }
  else{
    MicroPrintf("Bias type %s (%d) not supported.",
                TfLiteTypeGetName(bias->type), bias->type);
    return kTfLiteError;    
  }
#else
  return ConvReferenceEvalInt16(context, node);
#endif
}

TfLiteStatus EvalFloat32(TfLiteContext* context, TfLiteNode* node) {
#if defined(INCLUDE_FLOAT_OPT) && (defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kConvBiasTensor);

  return ConvEvalHifiFloat32(context, node, params, op_data, input, filter, bias,
                           output);
#else
  return ConvReferenceEvalFloat32(context, node);
#endif
}

}  // namespace

TFLMRegistration Register_CONV_2D_INT8() {
  return tflite::micro::RegisterOp(ConvInitXtensa, ConvPrepareXtensa, EvalInt8);
}

TFLMRegistration Register_CONV_2D_INT16() {
  return tflite::micro::RegisterOp(ConvInitXtensa, ConvPrepareXtensa,
                                   EvalInt16);
}

TFLMRegistration Register_CONV_2D_FLOAT32() {
  return tflite::micro::RegisterOp(ConvInitXtensa, ConvPrepareXtensa,
                                   EvalFloat32);
}

}  // namespace tflite
