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
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_pooling.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

TfLiteStatus AverageEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  switch (input->type) {
    case kTfLiteInt8: {
#if defined(HIFI5)
      auto* op_data = static_cast<const XtensaOpDataPooling*>(node->user_data);
      AverageEvalQuantizedHifi(context, node, params, op_data, input, output);
#elif defined(VISION_P6)
      const auto& op_data =
          *(reinterpret_cast<XtensaOpDataPooling*>(node->user_data));
      PoolEvalVision(context, node, *params, op_data, input, output);
#else
      const OpDataPooling* reference_op_data =
          static_cast<const OpDataPooling*>(node->user_data);
      AveragePoolingEvalQuantized<int8_t>(context, node, params,
                                          reference_op_data, input, output);
#endif
      break;
    }
    default: {
      MicroPrintf("Input type %s is not currently supported",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  switch (input->type) {
    case kTfLiteInt8: {
#if defined(HIFI5)
      auto* op_data = static_cast<const XtensaOpDataPooling*>(node->user_data);
      MaxEvalQuantizedHifi(context, node, params, op_data, input, output);
#elif defined(VISION_P6)
      const auto& op_data =
          *(reinterpret_cast<XtensaOpDataPooling*>(node->user_data));
      PoolEvalVision(context, node, *params, op_data, input, output);
#else
      const OpDataPooling* reference_op_data =
          static_cast<const OpDataPooling*>(node->user_data);
      MaxPoolingEvalQuantized<int8_t>(context, node, params, reference_op_data,
                                      input, output);
#endif
      break;
    }
    default: {
      MicroPrintf("Type %s not currently supported.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

#if defined(HIFI5)

TfLiteStatus AveragePrepareHifi(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(PoolingPrepare(context, node));
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kPoolingInputTensor);

  if (input->type == kTfLiteInt8) {
    const RuntimeShape& input_shape = GetTensorShape(input);
    TfLiteTensor* output =
        micro_context->AllocateTempInputTensor(node, kPoolingOutputTensor);
    const RuntimeShape& output_shape = GetTensorShape(output);
    micro_context->DeallocateTempTfLiteTensor(output);

    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
    auto* data = static_cast<XtensaOpDataPooling*>(node->user_data);

    int required_scratch = xa_nn_avgpool_getsize(
        depth, PREC_8, PREC_8, input_height, input_width, params->filter_height,
        params->filter_width,
        params->stride_width,                    // x_stride,
        params->stride_height,                   // y_stride,
        data->reference_op_data.padding.width,   // x_padding,
        data->reference_op_data.padding.height,  // y_padding,
        output_height, output_width, 0 /*NHWC input */, 0 /* NHWC output */);

    if (required_scratch <= 0) {
      MicroPrintf("Averagepool: xa_nn_avgpool_getsize failed");
      return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, required_scratch, &(data->scratch_tensor_index)));
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  return kTfLiteOk;
}

TfLiteStatus AverageEvalQuantizedHifi(TfLiteContext* context,
                                      const TfLiteNode* node,
                                      const TfLitePoolParams* params,
                                      const XtensaOpDataPooling* data,
                                      const TfLiteEvalTensor* input,
                                      TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteInt8);

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  void* p_scratch = static_cast<void*>(
      context->GetScratchBuffer(context, data->scratch_tensor_index));

  const int8_t* inp_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
  int8_t* out_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

  for (int batch = 0; batch < batches; ++batch) {
    TF_LITE_ENSURE_EQ(
        context,
        xa_nn_avgpool_8(
            &out_data_ptr[output_height * output_width * depth * batch],
            const_cast<int8_t*>(
                &inp_data_ptr[output_height * output_width * depth * batch]),
            input_height, input_width, depth, params->filter_height,
            params->filter_width, params->stride_width, params->stride_height,
            data->reference_op_data.padding.width,
            data->reference_op_data.padding.height, output_height, output_width,
            0, 0, p_scratch),
        0);
  }

  const int out_length = batches * output_height * output_width * depth;
  TF_LITE_ENSURE_EQ(
      context,
      xa_nn_vec_activation_min_max_8_8(
          out_data_ptr, out_data_ptr, data->reference_op_data.activation_min,
          data->reference_op_data.activation_max, out_length),
      0);

  return kTfLiteOk;
}

TfLiteStatus MaxPrepareHifi(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(PoolingPrepare(context, node));

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kPoolingInputTensor);

  if (input->type == kTfLiteInt8) {
    auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
    auto* data = static_cast<XtensaOpDataPooling*>(node->user_data);

    const RuntimeShape& input_shape = GetTensorShape(input);
    TfLiteTensor* output =
        micro_context->AllocateTempOutputTensor(node, kPoolingOutputTensor);
    const RuntimeShape& output_shape = GetTensorShape(output);
    micro_context->DeallocateTempTfLiteTensor(output);

    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    int required_scratch = xa_nn_maxpool_getsize(
        depth, PREC_8, PREC_8, input_height, input_width, params->filter_height,
        params->filter_width,
        params->stride_width,                    // x_stride,
        params->stride_height,                   // y_stride,
        data->reference_op_data.padding.width,   // x_padding,
        data->reference_op_data.padding.height,  // y_padding,
        output_height, output_width, 0 /* NHWC inpput */, 0 /* NHWC output */);

    if (required_scratch <= 0) {
      MicroPrintf("Maxpool: xa_nn_maxpool_getsize failed");
      return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, required_scratch, &(data->scratch_tensor_index)));
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  return kTfLiteOk;
}

TfLiteStatus MaxEvalQuantizedHifi(TfLiteContext* context, TfLiteNode* node,
                                  TfLitePoolParams* params,
                                  const XtensaOpDataPooling* data,
                                  const TfLiteEvalTensor* input,
                                  TfLiteEvalTensor* output) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  void* p_scratch = static_cast<void*>(
      context->GetScratchBuffer(context, data->scratch_tensor_index));

  const int8_t* inp_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
  int8_t* out_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

  for (int batch = 0; batch < batches; ++batch) {
    TF_LITE_ENSURE_EQ(
        context,
        xa_nn_maxpool_8(
            &out_data_ptr[output_height * output_width * depth * batch],
            const_cast<int8_t*>(
                &inp_data_ptr[output_height * output_width * depth * batch]),
            input_height, input_width, depth, params->filter_height,
            params->filter_width, params->stride_width, params->stride_height,
            data->reference_op_data.padding.width,
            data->reference_op_data.padding.height, output_height, output_width,
            0, 0, p_scratch),
        0);
  }

  const int out_length = batches * output_height * output_width * depth;
  TF_LITE_ENSURE_EQ(
      context,
      xa_nn_vec_activation_min_max_8_8(
          out_data_ptr, out_data_ptr, data->reference_op_data.activation_min,
          data->reference_op_data.activation_max, out_length),
      0);

  return kTfLiteOk;
}

#endif  // defined(HIFI5)

void* XtensaPoolingInit(TfLiteContext* context, const char* buffer,
                        size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if defined(HIFI5)
  return context->AllocatePersistentBuffer(context,
                                           sizeof(XtensaOpDataPooling));
#elif defined(VISION_P6)
  if (InitXtensaContext()) {
    return nullptr;
  }
  return context->AllocatePersistentBuffer(context,
                                           sizeof(XtensaOpDataPooling));
#else
  return context->AllocatePersistentBuffer(context, sizeof(OpDataPooling));
#endif
}

TFLMRegistration Register_AVERAGE_POOL_2D_INT8() {
#if defined(HIFI5)
  return tflite::micro::RegisterOp(XtensaPoolingInit, AveragePrepareHifi,
                                   AverageEvalInt8);
#elif defined(VISION_P6)
  return tflite::micro::RegisterOp(XtensaPoolingInit, AvgPoolingPrepareVision,
                                   AverageEvalInt8);
#else
  return tflite::micro::RegisterOp(XtensaPoolingInit, PoolingPrepare,
                                   AverageEvalInt8);
#endif
}

TFLMRegistration Register_MAX_POOL_2D_INT8() {
#if defined(HIFI5)
  return tflite::micro::RegisterOp(XtensaPoolingInit, MaxPrepareHifi,
                                   MaxEvalInt8);
#elif defined(VISION_P6)
  return tflite::micro::RegisterOp(XtensaPoolingInit, MaxPoolingPrepareVision,
                                   MaxEvalInt8);
#else
  return tflite::micro::RegisterOp(XtensaPoolingInit, PoolingPrepare,
                                   MaxEvalInt8);
#endif
}

}  // namespace tflite
