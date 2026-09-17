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
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_fully_connected.h"

namespace tflite {

TfLiteStatus XtensaEvalFullyConnectedQuantizedInt8(
    TfLiteContext* context, TfLiteNode* node, const OpDataFullyConnected& data,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
#if !defined(VISION_P6)

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node,
                                              kFullyConnectedWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kFullyConnectedBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  const int32_t* bias_data =
#ifdef USE_TFLM_COMPRESSION
      tflite::micro::GetOptionalTensorData<int32_t>(
          micro_context, bias, bias_comp_td, data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
      tflite::micro::GetOptionalTensorData<int32_t>(bias);
#endif  // USE_TFLM_COMPRESSION

  const int8_t* filter_data =
#ifdef USE_TFLM_COMPRESSION
      tflite::micro::GetTensorData<int8_t>(
          micro_context, filter, weights_comp_td, data.weights_scratch_index);
#else   // USE_TFLM_COMPRESSION
      tflite::micro::GetTensorData<int8_t>(filter);
#endif  // USE_TFLM_COMPRESSION

  // P6 Vision will handle INT4 filters as a reference operation.
  // For all other architectures, unpack INT4 here.
#if (defined(HIFI5) && defined(NNLIB_HIFI5)) || defined(HIFI_IQ)
  void* p_scratch = (void *)0;
#endif

  if (filter->type == kTfLiteInt4) {
#if (defined(HIFI5) && defined(NNLIB_HIFI5)) || defined(HIFI_IQ)
    p_scratch = static_cast<void*>(
    context->GetScratchBuffer(context, data.filter_buffer_index));
#else    
    int8_t* unpacked_filter_data = static_cast<int8_t*>(
        context->GetScratchBuffer(context, data.filter_buffer_index));

    tflite::tensor_utils::UnpackDenseInt4IntoInt8(
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(filter).FlatSize(), unpacked_filter_data);
    filter_data = unpacked_filter_data;
#endif    
  }

#endif  // !defined(VISION_P6)

#if defined(HIFIMINI)
  FullyConnectedEvalHifimini(FullyConnectedParamsQuantized(data),
                             tflite::micro::GetTensorShape(input),
                             tflite::micro::GetTensorData<int8_t>(input),
                             tflite::micro::GetTensorShape(filter), filter_data,
                             tflite::micro::GetTensorShape(bias), bias_data,
                             tflite::micro::GetTensorShape(output),
                             tflite::micro::GetTensorData<int8_t>(output));
#elif defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
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
        xa_nn_matmul_v2_per_chan_sym8sxasym8s_asym8s(
            tflite::micro::GetTensorData<int8_t>(output),
            filter_data,
            tflite::micro::GetTensorData<int8_t>(input),
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
#if (defined(HIFI5) && defined(NNLIB_HIFI5)) || defined(HIFI_IQ)   
      if(filter->type == kTfLiteInt4){
          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_fully_connected_v2_asym4sxasym8s_asym8s(
                  tflite::micro::GetTensorData<int8_t>(output),
                  filter_data,
                  tflite::micro::GetTensorData<int8_t>(input),
                  bias_data, accum_depth, output_depth, op_params.input_offset,
                  op_params.weights_offset, op_params.output_multiplier,
                  op_params.output_shift, op_params.output_offset, p_scratch,
                  data.output_activation_min, data.output_activation_max, NULL),
              0);
      }
      else{
#endif       
          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_fully_connected_v2_asym8sxasym8s_asym8s(
                  tflite::micro::GetTensorData<int8_t>(output),
                  filter_data,
                  tflite::micro::GetTensorData<int8_t>(input),
                  bias_data, accum_depth, output_depth, op_params.input_offset,
                  op_params.weights_offset, op_params.output_multiplier,
                  op_params.output_shift, op_params.output_offset,
                  data.output_activation_min, data.output_activation_max, NULL),
              0);
#if (defined(HIFI5) && defined(NNLIB_HIFI5)) || defined(HIFI_IQ)       
      }
#endif    
    }
    else {
#if (defined(HIFI5) && defined(NNLIB_HIFI5)) || defined(HIFI_IQ)
      if(filter->type == kTfLiteInt4){
        for (int b = 0; b < num_batches; ++b) {
          int8_t *base_input = (int8_t *)tflite::micro::GetTensorData<int8_t>(input);
          int8_t *input_ptr =  base_input + (b*accum_depth);
          int8_t *base_output = (int8_t *)tflite::micro::GetTensorData<int8_t>(output);
          int8_t *output_ptr = base_output + (b*output_depth);
          TF_LITE_ENSURE_EQ(
          context,       
          xa_nn_fully_connected_v2_asym4sxasym8s_asym8s(
              output_ptr,
              filter_data,
              input_ptr,
              bias_data, accum_depth, output_depth, op_params.input_offset,
              op_params.weights_offset, op_params.output_multiplier,
              op_params.output_shift, op_params.output_offset, p_scratch,
              data.output_activation_min, data.output_activation_max, NULL),
          0);
        }        
      }
      else{
#endif        
          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_matmul_v2_asym8sxasym8s_asym8s(
                  tflite::micro::GetTensorData<int8_t>(output),
                  filter_data,
                  tflite::micro::GetTensorData<int8_t>(input),
                  bias_data, output_depth, accum_depth, accum_depth,
                  num_batches, accum_depth, output_depth, 1,
                  op_params.weights_offset, op_params.input_offset,
                  op_params.output_multiplier, op_params.output_shift,
                  op_params.output_offset, 
                  data.output_activation_min, data.output_activation_max, NULL),
              0);
#if (defined(HIFI5) && defined(NNLIB_HIFI5)) || defined(HIFI_IQ)            
      }
#endif    
    }
  }

#elif defined(VISION_P6)
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  const auto& op_data =
      *(reinterpret_cast<XtensaFullyConnectedOpData*>(node->user_data));
  FullyConnectedEvalVision(context, node, params, op_data, input, filter, bias,
                           output);
#else
  if (data.is_per_channel)
  {
    reference_integer_ops::FullyConnectedPerChannel(
        FullyConnectedParamsQuantized(data),
        data.per_channel_output_multiplier,
        reinterpret_cast<const int*>(data.per_channel_output_shift),
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias), bias_data,
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
  else
  {
    reference_integer_ops::FullyConnected(
        FullyConnectedParamsQuantized(data), tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter), filter_data,
        tflite::micro::GetTensorShape(bias), bias_data,
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

  return kTfLiteOk;
}

namespace {

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
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

  return XtensaEvalFullyConnectedQuantizedInt8(context, node, data, input,
                                               filter, bias, output);
}

}  // namespace

TFLMRegistration Register_FULLY_CONNECTED_INT8() {
  return tflite::micro::RegisterOp(XtensaInitFullyConnected,
                                   XtensaPrepareFullyConnected, EvalInt8);
}

}  // namespace tflite
