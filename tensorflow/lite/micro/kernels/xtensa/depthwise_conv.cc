
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

#include "tensorflow/lite/micro/kernels/depthwise_conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_depthwise_conv.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
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

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, DepthwiseConvPrepare(context, node));
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);

  // For int16 input, only fallback to the reference kernel is used
  // so there is no need to prepare the Hifi/Vision kernel.
  if (input->type == kTfLiteInt16) {
    micro_context->DeallocateTempTfLiteTensor(input);
    return kTfLiteOk;
  }
  micro_context->DeallocateTempTfLiteTensor(input);

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  TF_LITE_ENSURE_OK(context, DepthwiseConvPrepareHifi(context, node));
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

#if defined(VISION_P6)
  TF_LITE_ENSURE_OK(context, DepthwiseConvPrepareVision(context, node));
#endif  // VISION_P6
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  const auto& op_data =
      *(reinterpret_cast<XtensaDepthwiseConvOpData*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kDepthwiseConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kDepthwiseConvBiasTensor)
          : nullptr;

  TfLiteEvalTensor filter_int8 = tflite::micro::MakeUnpackedInt4Tensor(
      context, op_data.reference_op_data.filter_buffer_index, filter);

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* filter_comp_td =
      micro_context->GetTensorCompressionData(node,
                                              kDepthwiseConvWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kDepthwiseConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteInt8: {
      switch (filter_int8.type) {
        case kTfLiteInt8: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
          DepthwiseConvEvalHifi(context, node, params, op_data, input,
                                &filter_int8, bias, output);
#elif defined(VISION_P6)
          DepthwiseConvEvalVision(context, node, params, op_data, input,
                                  &filter_int8, bias, output);
#else
          reference_integer_ops::DepthwiseConvPerChannel(
              DepthwiseConvParamsQuantized(params, op_data.reference_op_data),
              op_data.reference_op_data.per_channel_output_multiplier,
              op_data.reference_op_data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(
                  micro_context, &filter_int8, filter_comp_td,
                  op_data.reference_op_data.weights_scratch_index),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(
                  micro_context, bias, bias_comp_td,
                  op_data.reference_op_data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(&filter_int8),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
          break;
        }
        default:
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt16: {
      switch (filter->type) {
        case kTfLiteInt8: {
          reference_integer_ops::DepthwiseConvPerChannel(
              DepthwiseConvParamsQuantized(params, op_data.reference_op_data),
              op_data.reference_op_data.per_channel_output_multiplier,
              op_data.reference_op_data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(
                  micro_context, &filter_int8, filter_comp_td,
                  op_data.reference_op_data.weights_scratch_index),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int64_t>(
                  micro_context, bias, bias_comp_td,
                  op_data.reference_op_data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(&filter_int8),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int64_t>(bias),
#endif  // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        default:
          MicroPrintf("Filter type %s (%d) for input type %s not supported.",
                      TfLiteTypeGetName(filter->type), filter->type,
                      TfLiteTypeGetName(input->type));
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

TFLMRegistration Register_DEPTHWISE_CONV_2D() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
