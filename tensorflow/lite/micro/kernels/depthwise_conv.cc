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
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

void* DepthwiseConvInit(TfLiteContext* context, const char* buffer,
                        size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}

TfLiteStatus DepthwiseConvEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  const OpDataConv& data = *(static_cast<const OpDataConv*>(node->user_data));

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

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* filter_comp_td =
      micro_context->GetTensorCompressionData(node,
                                              kDepthwiseConvWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kDepthwiseConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::DepthwiseConv(
          DepthwiseConvParamsFloat(params, data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(micro_context, filter,
                                              filter_comp_td,
                                              data.weights_scratch_index),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(
              micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
          tflite::tensor_utils::UnpackDenseInt4IntoInt8(
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(filter).FlatSize(),
              unpacked_filter_data);
          reference_integer_ops::DepthwiseConvPerChannel(
              DepthwiseConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter), unpacked_filter_data,
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        case kTfLiteInt8: {
          reference_integer_ops::DepthwiseConvPerChannel(
              DepthwiseConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                   filter_comp_td,
                                                   data.weights_scratch_index),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(
                  micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
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
    case kTfLiteInt16: {
      switch (filter->type) {
        case kTfLiteInt8: {
          reference_integer_ops::DepthwiseConvPerChannel(
              DepthwiseConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                   filter_comp_td,
                                                   data.weights_scratch_index),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int64_t>(
                  micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(filter),
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
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_DEPTHWISE_CONV_2D() {
  return tflite::micro::RegisterOp(DepthwiseConvInit, DepthwiseConvPrepare,
                                   DepthwiseConvEval);
}

}  // namespace tflite
