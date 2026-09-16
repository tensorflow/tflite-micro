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
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_fully_connected.h"

namespace tflite {

TfLiteStatus XtensaEvalFullyConnectedQuantizedInt16(
    TfLiteContext* context, TfLiteNode* node, const OpDataFullyConnected& data,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
 
  if(bias != nullptr && bias->type == kTfLiteInt32) {
    MicroPrintf("Bias type %s (%d) not supported.",
                TfLiteTypeGetName(bias->type), bias->type);
    return kTfLiteError;
  }
  const int64_t* bias_data =
      nullptr != bias ? tflite::micro::GetTensorData<int64_t>(bias) : nullptr;

#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int num_batches =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  const int output_depth =
      output_shape.Dims(output_shape.DimensionsCount() - 1);

  const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  FullyConnectedParams op_params = FullyConnectedParamsQuantized(data);
  if (data.is_per_channel)
  {
    TF_LITE_ENSURE_EQ(
        context,
        xa_nn_matmul_v2_per_chan_sym8sxsym16s_sym16s(
            tflite::micro::GetTensorData<int16_t>(output),
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorData<int16_t>(input),
            bias_data, output_depth, accum_depth, accum_depth,
            num_batches, accum_depth, output_depth, 1,
            op_params.input_offset, data.per_channel_output_multiplier,
            data.per_channel_output_shift, op_params.output_offset, 
            data.output_activation_min, data.output_activation_max, NULL),
        0);
  }
  else
  {
    if(num_batches == 1) {
        TF_LITE_ENSURE_EQ(
            context,
            xa_nn_fully_connected_v2_sym8sxsym16s_sym16s(
                tflite::micro::GetTensorData<int16_t>(output),
                tflite::micro::GetTensorData<int8_t>(filter),
                tflite::micro::GetTensorData<int16_t>(input),
                bias_data, accum_depth, output_depth,
                op_params.output_multiplier, op_params.output_shift,
                data.output_activation_min, data.output_activation_max, NULL),
            0);
    }
    else{
        TF_LITE_ENSURE_EQ(
            context,
            xa_nn_matmul_v2_sym8sxsym16s_sym16s(
                tflite::micro::GetTensorData<int16_t>(output),
                tflite::micro::GetTensorData<int8_t>(filter),
                tflite::micro::GetTensorData<int16_t>(input),
                bias_data, output_depth, accum_depth, accum_depth,
                num_batches, accum_depth, output_depth, 1,
                op_params.input_offset, op_params.output_multiplier, 
                op_params.output_shift, op_params.output_offset,
                data.output_activation_min, data.output_activation_max, NULL),
            0);    
    }
  }
#else
  if (data.is_per_channel)
  {
    reference_integer_ops::FullyConnectedPerChannel(
        FullyConnectedParamsQuantized(data),
        data.per_channel_output_multiplier,
        reinterpret_cast<const int*>(data.per_channel_output_shift),
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias), bias_data,
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output))
  }
  else
  {
    reference_integer_ops::FullyConnected(
        FullyConnectedParamsQuantized(data), tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias), bias_data,
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
  }
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return kTfLiteOk;
}

namespace {

TfLiteStatus EvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data =
      *(static_cast<const OpDataFullyConnected*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  return XtensaEvalFullyConnectedQuantizedInt16(context, node, data, input,
                                               filter, bias, output);
}

}  // namespace

TFLMRegistration Register_FULLY_CONNECTED_INT16() {
  return tflite::micro::RegisterOp(XtensaInitFullyConnected,
                                   XtensaPrepareFullyConnected, EvalInt16);
}

}  // namespace tflite
