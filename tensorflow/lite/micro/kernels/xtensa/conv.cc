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
      tflite::micro::GetEvalInput(context, node, kConvBiasTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
#ifdef USE_TFLM_COMPRESSION

      MicroContext* micro_context = GetMicroContext(context);

      const CompressionTensorData* weights_comp_td =
          micro_context->GetTensorCompressionData(node, kConvWeightsTensor);
      const CompressionTensorData* bias_comp_td =
          micro_context->GetTensorCompressionData(node, kConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, op_data.reference_op_data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(
              micro_context, filter, weights_comp_td,
              op_data.reference_op_data.weights_scratch_index),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(
              micro_context, bias, bias_comp_td,
              op_data.reference_op_data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt8: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      if (params.dilation_width_factor == 1 &&
          params.dilation_height_factor == 1) {
        return ConvEvalHifiInt8(context, node, params, op_data, input, filter,
                                bias, output);
      } else {
        return ConvReferenceEvalInt8(context, node);
      }
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
    }
    case kTfLiteInt16: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      // Note that int32 bias is not widely supported and might be risky (e.g.
      // http://b/262003750). As such, while we have a fallback to the reference
      // implementation, production use-cases should only have int64 bias.
      if (bias->type == kTfLiteInt32) {
        return ConvReferenceEvalInt16(context, node);
      } else {
        return ConvEvalHifiInt16(context, node, params, op_data, input, filter,
                                 bias, output);
      }
#else
      return ConvReferenceEvalInt16(context, node);
#endif
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
