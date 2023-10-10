/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_softmax.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
TfLiteStatus PrepareHifi(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, SoftmaxPrepare(context, node));

  MicroContext* micro_context = GetMicroContext(context);
  // Calculate scratch memory requirements and request scratch buffer
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);

  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& output_shape = GetTensorShape(output);
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  if (input->type == kTfLiteInt8) {
    int required_scratch =
        get_softmax_scratch_size(PREC_ASYM8S, PREC_ASYM8S, depth);
    TF_LITE_ENSURE(context, required_scratch > 0);

    auto* data = static_cast<XtensaSoftmaxOpData*>(node->user_data);
    TF_LITE_ENSURE_OK(
        context, context->RequestScratchBufferInArena(
                     context, required_scratch, &(data->scratch_tensor_index)));
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus EvalHifi(const XtensaSoftmaxOpData* op_data,
                      const TfLiteEvalTensor* input, TfLiteEvalTensor* output,
                      TfLiteContext* context) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  void* p_scratch = static_cast<void*>(
      context->GetScratchBuffer(context, op_data->scratch_tensor_index));

  for (int i = 0; i < outer_size; ++i) {
    int err = xa_nn_vec_softmax_asym8s_16(
        &output_data[i * depth], &input_data[i * depth],
        op_data->params.diff_min, op_data->params.input_left_shift,
        op_data->params.input_multiplier, depth, p_scratch);
    TF_LITE_ENSURE(context, err == 0);
  }
  return kTfLiteOk;
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

}  // namespace

void* XtensaInitSoftmax(TfLiteContext* context, const char* buffer,
                        size_t length) {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(XtensaSoftmaxOpData));
#elif defined(VISION_P6)
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  if (InitXtensaContext()) {
    return nullptr;
  }
  return context->AllocatePersistentBuffer(context,
                                           sizeof(XtensaSoftmaxOpData));
#else
  return SoftmaxInit(context, buffer, length);
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
}

TfLiteStatus XtensaPrepareSoftmax(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  return PrepareHifi(context, node);
#else
  TF_LITE_ENSURE_OK(context, SoftmaxPrepare(context, node));
#if defined(VISION_P6)
  TF_LITE_ENSURE_OK(context, SoftmaxPrepareVision(context, node));
#endif
  return kTfLiteOk;
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
}

TfLiteStatus XtensaEvalSoftmaxInt8Int16(TfLiteContext* context,
                                        TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TFLITE_DCHECK(node->user_data != nullptr);

  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt16) {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
    return EvalHifi(static_cast<XtensaSoftmaxOpData*>(node->user_data), input,
                    output, context);
#else
    SoftmaxParams op_data = *static_cast<SoftmaxParams*>(node->user_data);
    tflite::reference_ops::Softmax(
        op_data, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
    return kTfLiteOk;
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  } else {
    MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                input->type);
    return kTfLiteError;
  }
}

TFLMRegistration Register_SOFTMAX_INT8_INT16() {
  return tflite::micro::RegisterOp(XtensaInitSoftmax, XtensaPrepareSoftmax,
                                   XtensaEvalSoftmaxInt8Int16);
}

}  // namespace tflite
