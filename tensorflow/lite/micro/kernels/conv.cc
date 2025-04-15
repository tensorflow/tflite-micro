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

#if defined(TFLM_USE_RISCV_VECTOR)
#include "tensorflow/lite/micro/kernels/riscv_vector/conv_rvv.h"
#endif

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
#if defined(TFLM_USE_RISCV_VECTOR)
#ifdef USE_TFLM_COMPRESSION
          TFLITE_DCHECK(weights_comp_td == nullptr && bias_comp_td == nullptr);
          if (weights_comp_td != nullptr || bias_comp_td != nullptr) 
          {
              MicroPrintf("ERROR: RVV path does not support compressed weights/bias yet.");
              return kTfLiteError;
          }
#endif // USE_TFLM_COMPRESSION
          const TfLiteConvParams* conv_params_rvv = // Use different name to avoid shadowing
              static_cast<const TfLiteConvParams*>(node->builtin_data);
          const TfLiteEvalTensor* input_rvv =
              tflite::micro::GetEvalInput(context, node, kConvInputTensor);
          const TfLiteEvalTensor* filter_rvv =
              tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
          const TfLiteEvalTensor* bias_rvv =
              (NumInputs(node) == 3)
                  ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
                  : nullptr;
          TfLiteEvalTensor* output_rvv =
              tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

          if (bias_rvv != nullptr && bias_rvv->type != kTfLiteInt32) {
             MicroPrintf("RVV kernel requires Int32 bias, got %s", TfLiteTypeGetName(bias_rvv->type));
             return kTfLiteError;
          }

          const int8_t* input_data_ptr = tflite::micro::GetTensorData<int8_t>(input_rvv);
          const int8_t* filter_data_ptr = tflite::micro::GetTensorData<int8_t>(filter_rvv);
          const int32_t* bias_data_ptr =
              (bias_rvv) ? tflite::micro::GetTensorData<int32_t>(bias_rvv) : nullptr;
          int8_t* output_data_ptr = tflite::micro::GetTensorData<int8_t>(output_rvv);

          const int32_t input_zero_point_arg = data.input_zero_point;
          const int32_t output_zero_point_arg = data.output_zero_point;
          const int32_t* output_multiplier_ptr = data.per_channel_output_multiplier;
          const int32_t* output_shift_ptr = data.per_channel_output_shift;

          const uint16_t input_height = static_cast<uint16_t>(input_rvv->dims->data[1]);
          const uint16_t input_width = static_cast<uint16_t>(input_rvv->dims->data[2]);
          const uint16_t input_channels = static_cast<uint16_t>(input_rvv->dims->data[3]);

          const uint16_t filter_height = static_cast<uint16_t>(filter_rvv->dims->data[1]);
          const uint16_t filter_width = static_cast<uint16_t>(filter_rvv->dims->data[2]);

          const uint16_t output_channels = static_cast<uint16_t>(output_rvv->dims->data[3]);

          const uint16_t output_height = static_cast<uint16_t>(output_rvv->dims->data[1]);
          const uint16_t output_width = static_cast<uint16_t>(output_rvv->dims->data[2]);

          const uint16_t stride_height = static_cast<uint16_t>(conv_params_rvv->stride_height);
          const uint16_t stride_width = static_cast<uint16_t>(conv_params_rvv->stride_width);
          const uint16_t pad_height = static_cast<uint16_t>(data.padding.height);
          const uint16_t pad_width = static_cast<uint16_t>(data.padding.width);

          // Call the optimized RVV kernel
          convolution_hwc_ohwi_rvv(
            input_data_ptr,
            input_height,
            input_width,
            input_channels,
            input_zero_point_arg,
            filter_data_ptr,
            filter_height,
            filter_width,
            bias_data_ptr,
            output_data_ptr,
            output_height,
            output_width,
            output_channels,
            output_zero_point_arg,
            output_multiplier_ptr,
            output_shift_ptr,
            stride_height,
            stride_width,
            pad_height,
            pad_width,
            data.output_activation_min,
            data.output_activation_max,
            data.dilation_height_factor,
            data.dilation_width_factor
           );
#else // defined(TFLM_USE_RISCV_VECTOR)
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
#endif
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
