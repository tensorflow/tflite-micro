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

#include "tensorflow/lite/micro/kernels/depthwise_conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
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

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
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

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::DepthwiseConv(
          DepthwiseConvParamsFloat(params, op_data.reference_op_data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    }
    case kTfLiteInt8: {
#if defined(HIFI4) || defined(HIFI5) || defined(VISION_P6)
      if (op_data.can_optimize) {
        // optimized Xtensa code will unpack filter data if necessary
#if defined(HIFI4) || defined(HIFI5)
        return DepthwiseConvEvalHifi(context, node, params, op_data, input,
                                     filter, bias, output);
#elif defined(VISION_P6)
        return DepthwiseConvEvalVision(context, node, params, op_data, input,
                                       filter, bias, output);
#endif  // defined(HIFI4) || defined(HIFI5)
      }
#endif  // defined(HIFI4) || defined(HIFI5) || defined(VISION_P6)

      const int8_t* filter_data;
      if (filter->type == kTfLiteInt4) {
        int8_t* unpacked_filter_data =
            static_cast<int8_t*>(context->GetScratchBuffer(
                context, op_data.reference_op_data.filter_buffer_index));
        tflite::tensor_utils::UnpackDenseInt4IntoInt8(
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(filter).FlatSize(),
            unpacked_filter_data);
        filter_data = unpacked_filter_data;
      } else {
        filter_data = tflite::micro::GetTensorData<int8_t>(filter);
      }

      reference_integer_ops::DepthwiseConvPerChannel(
          DepthwiseConvParamsQuantized(params, op_data.reference_op_data),
          op_data.reference_op_data.per_channel_output_multiplier,
          op_data.reference_op_data.per_channel_output_shift,
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter), filter_data,
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<int32_t>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));

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
  return tflite::micro::RegisterOp(DepthwiseConvInitXtensa,
                                   DepthwiseConvPrepareXtensa, Eval);
}

}  // namespace tflite
