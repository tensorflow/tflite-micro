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
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node, kConvWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data), tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(micro_context, filter,
                                              weights_comp_td,
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
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt16: {
      if (bias == nullptr || bias->type == kTfLiteInt32) {
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, data),
            data.per_channel_output_multiplier, data.per_channel_output_shift,
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                 weights_comp_td,
                                                 data.weights_scratch_index),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<int32_t>(
                micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<std::int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      } else if (bias->type == kTfLiteInt64) {
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, data),
            data.per_channel_output_multiplier, data.per_channel_output_shift,
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                 weights_comp_td,
                                                 data.weights_scratch_index),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetTensorData<int64_t>(
                micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetTensorData<std::int64_t>(bias),
#endif  // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      } else {
        MicroPrintf("Bias type %s (%d) not supported.",
                    TfLiteTypeGetName(bias->type), bias->type);
        return kTfLiteError;
      }
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
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
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
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                   weights_comp_td,
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
          MicroPrintf("Weight type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
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
  return tflite::micro::RegisterOp(ConvInit, ConvPrepare, ConvEval);
}

}  // namespace tflite
