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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

TfLiteStatus ConvReferenceEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  const auto& op_data = *(reinterpret_cast<OpDataConv*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node, kConvWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  const int8_t* filter_data;
  if (filter->type == kTfLiteInt4) {
    int8_t* unpacked_filter_data = static_cast<int8_t*>(
        context->GetScratchBuffer(context, op_data.filter_buffer_index));
    tflite::tensor_utils::UnpackDenseInt4IntoInt8(
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(filter).FlatSize(), unpacked_filter_data);
    filter_data = unpacked_filter_data;
  } else {
#ifdef USE_TFLM_COMPRESSION
    filter_data = tflite::micro::GetTensorData<int8_t>(
        micro_context, filter, weights_comp_td, op_data.weights_scratch_index);
#else   // USE_TFLM_COMPRESSION
    filter_data = tflite::micro::GetTensorData<int8_t>(filter);
#endif  // USE_TFLM_COMPRESSION
  }

  reference_integer_ops::ConvPerChannel(
      ConvParamsQuantized(params, op_data),
      op_data.per_channel_output_multiplier, op_data.per_channel_output_shift,
      tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter), filter_data,
      tflite::micro::GetTensorShape(bias),
#ifdef USE_TFLM_COMPRESSION
      tflite::micro::GetOptionalTensorData<int32_t>(
          micro_context, bias, bias_comp_td, op_data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
      tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));

  return kTfLiteOk;
}

// TODO(b/189981943): This variant can be used for a smaller binary
// since the optimized conv implementation currently adds a lot to
// the binary size (~30KB to text section).
TFLMRegistration Register_CONV_2D_INT8REF() {
  return tflite::micro::RegisterOp(ConvInit, ConvPrepare,
                                   ConvReferenceEvalInt8);
}

}  // namespace tflite
