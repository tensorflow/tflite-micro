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

#include <cstddef>
#include <cstdint>
#include <new>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_softmax.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace tflite {
namespace micro {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
TfLiteStatus PrepareHifi(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, SoftmaxPrepare(context, node));

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kSoftmaxInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kSoftmaxOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& output_shape = GetTensorShape(output);
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  auto* data = static_cast<XtensaSoftmaxOpData*>(node->user_data);
  data->softmax_lut = nullptr;

  if (input->type == kTfLiteInt8) {
    if (output->type == kTfLiteInt8 &&
        (outer_size * depth) > kInt8SoftmaxLutThreshold) {
      const int depth_aligned = (depth + 7) & ~7;
      const int required_scratch = 16 * depth_aligned;
      TF_LITE_ENSURE(context, required_scratch > 0);
      TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                     context, required_scratch,
                                     &(data->scratch_tensor_index)));

      data->softmax_lut =
          static_cast<int32_t*>(context->AllocatePersistentBuffer(
              context, kInt8SoftmaxLutSize * sizeof(int32_t)));
      TF_LITE_ENSURE(context, data->softmax_lut != nullptr);
      TF_LITE_ENSURE_EQ(
          context,
          xa_nn_init_lut_asym8s_softmax(
              data->softmax_lut, data->params.diff_min,
              data->params.input_multiplier, data->params.input_left_shift),
          0);
    } else {
      const int out_precision =
          (output->type == kTfLiteInt16) ? PREC_SYM16S : PREC_ASYM8S;
      const int required_scratch =
          get_softmax_scratch_size(PREC_ASYM8S, out_precision, depth);
      TF_LITE_ENSURE(context, required_scratch > 0);
      TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                     context, required_scratch,
                                     &(data->scratch_tensor_index)));
    }
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus EvalHifi(const XtensaSoftmaxOpData& op_data,
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

  void* p_scratch =
      context->GetScratchBuffer(context, op_data.scratch_tensor_index);
  if (p_scratch == nullptr) {
    return kTfLiteError;
  }

  for (int i = 0; i < outer_size; ++i) {
    int err = xa_nn_vec_softmax_asym8s_16(
        &output_data[i * depth], &input_data[i * depth],
        op_data.params.diff_min, op_data.params.input_left_shift,
        op_data.params.input_multiplier, depth, p_scratch);
    if (err != 0) {
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)

}  // namespace

void* XtensaInitSoftmax(TfLiteContext* context, const char* buffer,
                        size_t length) {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data =
      context->AllocatePersistentBuffer(context, sizeof(XtensaSoftmaxOpData));
  return new (data) XtensaSoftmaxOpData();
#elif defined(VISION_P6)
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  if (InitXtensaContext()) {
    return nullptr;
  }
  void* data =
      context->AllocatePersistentBuffer(context, sizeof(XtensaSoftmaxOpData));
  return new (data) XtensaSoftmaxOpData();
#else
  return SoftmaxInit(context, buffer, length);
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)
}

TfLiteStatus XtensaPrepareSoftmax(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return PrepareHifi(context, node);
#else
  TF_LITE_ENSURE_OK(context, SoftmaxPrepare(context, node));
#if defined(VISION_P6)
  TF_LITE_ENSURE_OK(context, SoftmaxPrepareVision(context, node));
#endif
  return kTfLiteOk;
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)
}

TfLiteStatus XtensaEvalSoftmaxInt8Int16(TfLiteContext* context,
                                        TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSoftmaxInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSoftmaxOutputTensor);
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(input->type == kTfLiteInt8);
  TFLITE_DCHECK(output->type == kTfLiteInt16);

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return EvalHifi(*static_cast<const XtensaSoftmaxOpData*>(node->user_data),
                  input, output, context);
#else
  tflite::micro::reference_ops::Softmax(
      XtensaGetSoftmaxParams(node), tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int16_t>(output));
  return kTfLiteOk;
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5) ||
        // defined(HIFI_IQ)
}

TFLMRegistration Register_SOFTMAX_INT8_INT16() {
  return tflite::micro::RegisterOp(
      XtensaInitSoftmax, XtensaPrepareSoftmaxTyped<kTfLiteInt8, kTfLiteInt16>,
      XtensaEvalSoftmaxInt8Int16);
}

}  // namespace micro
}  // namespace tflite
